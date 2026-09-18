import json, argparse, time
from pathlib import Path
from obstacle_constraint import DepthObstacleConstraint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trials',type=int,default=20); ap.add_argument('--log',default='outputs/vlm_planning/safety_ablation.json'); a=ap.parse_args(); gate=DepthObstacleConstraint(.1); rows=[]
    for method in ('unprotected_sim','depth_guard'):
        success=stops=hazards=0
        for i in range(a.trials):
            traj=[[0,0,.3],[0,0,.05]] if i%4==0 else [[0,0,.3],[0,0,.2]]
            check=gate.evaluate_trajectory(traj,[[0,0,.05]])
            if method=='depth_guard' and check['stop']: stops+=1
            elif method=='unprotected_sim' and check['stop']: hazards+=1; success+=1
            else: success+=1
        rows.append({'method':method,'trials':a.trials,'successes':success,'success_rate':success/a.trials,'safety_stops':stops,'hazard_events':hazards})
    out=Path(a.log); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(rows,ensure_ascii=False,indent=2)); print('日志已保存:',out)
if __name__=='__main__': main()
