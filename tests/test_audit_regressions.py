"""CPU-only regression tests. No models, datasets, cameras or robot connections."""
import ast
import hashlib
import copy
import math
import random
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from franka_teleop.camera_frames import validated_frame_pair
from franka_teleop.pi05_engine.contracts import (
    resolve_crop, checkpoint_state, validate_resume, lora_spec, ACTION_SEMANTICS, PRECISION)
from franka_teleop.pi05_engine.lora import load_lora_state_dict
from franka_teleop import closed_loop_franka as rtc

def extract(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]

class Contracts(unittest.TestCase):
    def test_metadata_wins_over_directory_and_rename(self):
        self.assertEqual(resolve_crop({"image_crop": "none"}, "pi05_lora_crop169_20k/x.pt"), "none")
        self.assertEqual(resolve_crop({"image_crop": "16_9"}, "renamed/x.pt"), "16_9")
        with self.assertRaises(ValueError):
            resolve_crop({"image_crop": "16_9"}, "x.pt", "none")

    def test_legacy_is_explicit_not_substring(self):
        with self.assertWarns(UserWarning):
            self.assertEqual(resolve_crop({}, "pi05_lora_pure_flow_50k/x.pt"), "none")
        with self.assertWarns(UserWarning):
            self.assertEqual(resolve_crop({}, "pi05_lora_crop169_20k/x.pt"), "16_9")
        with self.assertRaises(ValueError):
            resolve_crop({}, "copied_crop169_unknown/x.pt")
        self.assertEqual(resolve_crop({}, "x.pt", "none"), "none")

    def test_invalid_state_and_rank(self):
        with self.assertRaises(ValueError):
            checkpoint_state({"step": 5000})
        self.assertEqual(lora_spec({"lang_rank": 4, "expert_rank": 8})["lang_alpha"], 8)
        with self.assertRaises(ValueError):
            lora_spec({"lang_rank": 0})

    def resume_fixture(self):
        args = types.SimpleNamespace(resume_mode="strict", image_crop="none",
            lang_rank=16, expert_rank=32, lora_dropout=.05, dataset=[ROOT / "dataset/example"],
            task_filter=None, val_seed=42, num_val_episodes=4, val_ratio=.15,
            lr=.0001, steps=20000, warmup_steps=200, batch_size=4, grad_accum=2)
        ckpt = dict(image_crop="none", action_semantics=ACTION_SEMANTICS, precision=PRECISION,
                    val_episodes=["ep1"], optimizer_state_dict={}, scheduler_state_dict={},
                    step=2500, args=vars(args).copy(), state_dict={"w":torch.ones(2)})
        return ckpt, args

    def test_resume_contract_rejects_silent_changes(self):
        ckpt, args = self.resume_fixture()
        validate_resume(ckpt, args, ["ep1"])
        for key, bad in [("image_crop","16_9"), ("action_semantics","old"),
                         ("precision","bf16"), ("val_episodes",["ep2"])]:
            c=copy.deepcopy(ckpt); c[key]=bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume(c,args,["ep1"])
        args.lr = .0002
        with self.assertRaises(ValueError):
            validate_resume(ckpt,args)

    def test_legacy_requires_explicit_migration(self):
        args=self.resume_fixture()[1]
        old={"state_dict":{"w":torch.ones(2,dtype=torch.bfloat16)}}
        with self.assertRaises(ValueError):
            validate_resume(old,args)
        args.resume_mode="migrate"
        validate_resume(old,args)
        with self.assertRaises(ValueError):
            validate_resume({"step":5},args)

class AtomicLoader(unittest.TestCase):
    def setUp(self):
        self.net=torch.nn.Linear(2,2)
        self.original={k:v.detach().clone() for k,v in self.net.named_parameters()}

    def assert_unchanged(self):
        for k,v in self.net.named_parameters():
            self.assertTrue(torch.equal(v,self.original[k]),k)

    def test_missing_and_extra_leave_model_unchanged(self):
        for state in [{"weight":torch.full((2,2),9.)},
                      {"weight":torch.full((2,2),9.),"bias":torch.ones(2),"extra":torch.ones(1)}]:
            with self.assertRaises(KeyError):
                load_lora_state_dict(self.net,state,strict=True)
            self.assert_unchanged()

    def test_broadcast_and_nan_rejected_before_copy(self):
        for bias in [torch.ones(1), torch.tensor([float("nan"),1.])]:
            with self.assertRaises(ValueError):
                load_lora_state_dict(self.net,{"weight":torch.full((2,2),9.),"bias":bias},strict=True)
            self.assert_unchanged()

    def test_fp32_values_preserved(self):
        desired={k:torch.full_like(v,1.00001) for k,v in self.original.items()}
        load_lora_state_dict(self.net,desired,strict=True)
        self.assertTrue(torch.equal(self.net.weight,desired["weight"]))

    def test_copy_failure_rolls_back(self):
        original_copy=torch.Tensor.copy_
        calls=[0]
        def fail_once(target, source, *a, **kw):
            calls[0]+=1
            if calls[0]==2:
                raise RuntimeError("simulated copy failure")
            return original_copy(target,source,*a,**kw)
        with patch.object(torch.Tensor,"copy_",fail_once), self.assertRaises(RuntimeError):
            load_lora_state_dict(self.net,{k:torch.full_like(v,9.) for k,v in self.original.items()},strict=True)
        self.assert_unchanged()

class Cameras(unittest.TestCase):
    def test_fresh_copied_and_bad_pairs_rejected(self):
        frames={r:np.zeros((2,2,3),np.uint8) for r in ("front","wrist")}
        f,w=validated_frame_pair(frames,{"front":99.98,"wrist":99.97},100)
        self.assertIsNotNone(f); self.assertIsNot(f,frames["front"])
        for stamps in [{"front":0.,"wrist":0.}, {"front":99.99,"wrist":99.6},
                       {"front":101.,"wrist":100.}, {"front":float("nan"),"wrist":100.}]:
            self.assertEqual(validated_frame_pair(frames,stamps,100),(None,None))

    def test_both_actual_entry_methods(self):
        for file in ("sync_vla_agent.py","async_rtc_vla_agent.py"):
            method=extract("scripts/python/"+file,"get_frames",
                {"time":types.SimpleNamespace(monotonic=lambda:100.),
                 "validated_frame_pair":validated_frame_pair})
            obj=types.SimpleNamespace(lock=threading.Lock(),
                frames={r:np.zeros(1) for r in ("front","wrist")},
                timestamps={"front":0.,"wrist":0.},
                last_valid_frames={r:np.ones(1) for r in ("front","wrist")})
            self.assertEqual(method(obj),(None,None))
            obj.timestamps={"front":99.99,"wrist":99.6}
            self.assertEqual(method(obj),(None,None))

class Timing(unittest.TestCase):
    def chunk(self,stamp=0.,rid=1):
        return {"_requested_at":stamp,"_request_id":rid,"joint_positions":[[0.]*7]*15}

    def test_elapsed_clock_and_expiry(self):
        c=self.chunk()
        self.assertEqual(rtc.chunk_start_index(c,.8),12)
        self.assertIsNone(rtc.chunk_start_index(c,1.01))
        self.assertIsNone(rtc.chunk_start_index(c,.1,last_request_id=1))
        self.assertIsNone(rtc.chunk_start_index(dict(c,hold=True),.1))
        self.assertIsNone(rtc.chunk_start_index(c,-1.))
        self.assertIsNone(rtc.chunk_start_index({"joint_positions":[[0.]*7]},.1))

    def test_starvation_arrival_uses_same_activation(self):
        # Drive the actual control loop with a virtual clock, no sockets/hardware.
        clock=types.SimpleNamespace(now=0.)
        q=np.array([.13,.48,.05,-2.45,.02,2.9,.94])
        commands=[]; stops=[]
        arm=types.SimpleNamespace(
            get_gripper_width=lambda:.08,
            get_cartesian_pose=lambda:(np.array([.4,0,.4]),None),
            get_joint_positions=lambda:q.copy(),
            set_joint_velocities=lambda v:commands.append((clock.now,v.copy())),
            stop=lambda:stops.append(clock.now))
        chunks=self
        class Client:
            def __init__(self,conn): self.count=0; self.pending=None
            def request_chunk(self,msg):
                self.count+=1
                c=chunks.chunk(clock.now,self.count)
                c["joint_positions"]=[(q+np.array([.001*(k+1),0,0,0,0,0,0])).tolist() for k in range(15)]
                c["gripper"]=[0.]*15
                self.pending=c
                return True
            def get_chunk(self,block=False,timeout=None):
                if self.pending is None: return None
                if self.count==1:
                    c=self.pending; self.pending=None; return c
                if block:
                    clock.now+=.02
                    c=self.pending; self.pending=None; return c
                return None
            def is_alive(self): return True
            def stop(self): pass
        args=types.SimpleNamespace(preempt_step=8,steps_per_chunk=8,z_min=-1.,
              max_vel=10.,kp_pos=1.,flip_lr=False,shadow=False,close_delay_steps=2)
        fake_ros=types.SimpleNamespace(
            Rate=lambda hz:types.SimpleNamespace(sleep=lambda:setattr(clock,"now",clock.now+1/hz)),
            is_shutdown=lambda:clock.now>1.19)
        with patch.object(rtc,"AsyncChunkClient",Client), patch.object(rtc,"rospy",fake_ros), \
             patch.object(rtc.time,"monotonic",lambda:clock.now), \
             patch.object(rtc,"project_velocity_z_floor",lambda q,v,**kw:(v,False)):
            rtc.run_rtc_loop(arm,None,args)
        after=[v[0] for t,v in commands if t>1.05]
        self.assertTrue(after,"No post-starvation command executed")
        self.assertGreater(after[0],.0015,"New chunk replayed index zero")
        self.assertTrue(any(t>=.99 for t in stops),"Arm was not stopped before waiting")

class Labels(unittest.TestCase):
    def test_train_eval_use_command_t_without_one_frame_shift(self):
        states=np.zeros((20,8),np.float32); states[:,0]=np.arange(20)
        states[:,7]=.7
        actions=np.zeros((20,8),np.float32); actions[3:,7]=1.
        frames=np.zeros((20,2,2,3),np.uint8)
        cf=dict(states=states,actions=actions,front_frames=frames,wrist_frames=frames)
        sample=dict(file_key="f",frame_idx=2,task="pick")
        ns=dict(torch=torch,np=np,hashlib=hashlib,CHUNK_SIZE=15,random=random,cached_files={"f":cf},
                train_samples=[sample],task_names=["pick"],
                task_samplers={"pick":types.SimpleNamespace(next_idx=lambda:0)})
        batch=extract("scripts/python/train_pi05_lora.py","sample_balanced_batch",ns)(1)
        expected=np.column_stack([states[3:18,:7]-states[2,:7],actions[2:17,7]])
        np.testing.assert_array_equal(batch[3][0],expected)
        seen=[]
        def predict(obs,task,noise):
            seen.append(noise.clone())
            return torch.from_numpy(expected[None])
        inference=types.SimpleNamespace(predict_action_chunk=predict,
            config=types.SimpleNamespace(chunk_size=15,max_action_dim=32))
        result=extract("scripts/python/train_pi05_lora.py","evaluate_sample",ns)(inference,sample,{"f":cf})
        self.assertEqual(result["gripper_mse"],0.)
        self.assertEqual(result["action_mse"],0.)
        rng_before=torch.random.get_rng_state().clone()
        ns["evaluate_sample"](inference,sample,{"f":cf})
        self.assertTrue(torch.equal(seen[0],seen[1]))
        self.assertTrue(torch.equal(rng_before,torch.random.get_rng_state()))


class HttpCheckpoint(unittest.TestCase):
    def fixture(self):
        ns=dict(threading=threading, torch=torch, np=np, Path=Path,
                read_checkpoint=lambda p:self.checkpoint, checkpoint_state=checkpoint_state,
                resolve_crop=resolve_crop,lora_spec=lora_spec,load_lora_state_dict=load_lora_state_dict)
        cls=extract("scripts/python/vla_inference_server.py","VLAEngine",ns)
        engine=cls.__new__(cls)
        net=torch.nn.Module()
        net.lora_A=torch.nn.Parameter(torch.zeros(2,2))
        net.action_out_proj=torch.nn.Linear(2,2)
        engine.device="cpu"; engine.lock=threading.Lock(); engine.image_crop="auto"
        engine._lora_spec=lora_spec({})
        engine.current_ckpt_info={"name":"original","image_crop":"none"}
        engine.inference=types.SimpleNamespace(network=net,
            processor=types.SimpleNamespace(crop_mode="none"),reset=lambda:None)
        # An existing source file suffices; read_checkpoint is an in-memory fixture.
        engine.resolve_checkpoint_path=lambda key:ROOT/"tests/test_audit_regressions.py"
        self.checkpoint={"image_crop":"16_9","state_dict":{k:torch.full_like(v,1.00001) for k,v in net.named_parameters()}}
        return engine

    def test_switch_updates_crop_and_keeps_fp32(self):
        engine=self.fixture()
        result=engine.load_checkpoint("new")
        self.assertEqual(result["status"],"success")
        self.assertEqual(engine.inference.processor.crop_mode,"16_9")
        self.assertTrue(torch.equal(engine.inference.network.action_out_proj.weight,
                                   self.checkpoint["state_dict"]["action_out_proj.weight"]))

    def test_failed_switch_preserves_weights_crop_and_identity(self):
        engine=self.fixture()
        before={k:v.detach().clone() for k,v in engine.inference.network.named_parameters()}
        self.checkpoint["state_dict"]["action_out_proj.bias"]=torch.zeros(1)
        with self.assertRaises(ValueError):
            engine.load_checkpoint("bad")
        for k,v in engine.inference.network.named_parameters():
            self.assertTrue(torch.equal(v,before[k]))
        self.assertEqual(engine.inference.processor.crop_mode,"none")
        self.assertEqual(engine.current_ckpt_info["name"],"original")

    def test_rank_change_rejected_before_mutation(self):
        engine=self.fixture()
        self.checkpoint["lang_rank"]=8
        with self.assertRaises(ValueError):
            engine.load_checkpoint("different-rank")
        self.assertEqual(engine.current_ckpt_info["name"],"original")

class EntryContracts(unittest.TestCase):
    def test_all_deploy_entries_preserve_fp32_projections(self):
        for name in ("sync_vla_agent.py","async_rtc_vla_agent.py","vla_inference_server.py"):
            tree=ast.parse((ROOT/"scripts/python"/name).read_text(encoding="utf-8"))
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Attribute) and n.func.attr=="from_checkpoint"]
            self.assertTrue(calls)
            for call in calls:
                kwargs={k.arg:k.value for k in call.keywords}
                self.assertIs(kwargs["trainable_fp32"].value,True)

    def test_resume_restores_scheduler_learning_rate(self):
        p=torch.nn.Parameter(torch.ones(2))
        opt=torch.optim.AdamW([p],lr=.001)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1./(step+1))
        for _ in range(3):
            p.grad=torch.ones_like(p); opt.step(); scheduler.step()
        saved_opt=copy.deepcopy(opt.state_dict())
        saved_scheduler=copy.deepcopy(scheduler.state_dict())
        q=torch.nn.Parameter(p.detach().clone())
        restored=torch.optim.AdamW([q],lr=.001)
        restored.load_state_dict(saved_opt)
        restored_scheduler=torch.optim.lr_scheduler.LambdaLR(restored,lambda step:1./(step+1))
        restored_scheduler.load_state_dict(saved_scheduler)
        for group, old in zip(restored.param_groups,saved_opt["param_groups"]):
            group["lr"]=old["lr"]
        q.grad=torch.ones_like(q); p.grad=torch.ones_like(p)
        restored.step(); restored_scheduler.step()
        opt.step(); scheduler.step()
        self.assertTrue(torch.equal(q,p))
        self.assertEqual(restored.param_groups[0]["lr"],opt.param_groups[0]["lr"])

if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
