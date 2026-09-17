import numpy as np
try:
    from .coordinate_transform import pixel_to_camera
except ImportError:
    from coordinate_transform import pixel_to_camera

def robust_depth_m(depth_image, u, v, depth_scale=0.001, radius=2):
    a=np.asarray(depth_image); h,w=a.shape[:2]; u=int(round(u)); v=int(round(v))
    x0,x1=max(0,u-radius),min(w,u+radius+1); y0,y1=max(0,v-radius),min(h,v+radius+1)
    vals=a[y0:y1,x0:x1].astype(float).ravel(); vals=vals[np.isfinite(vals) & (vals>0)]
    if not len(vals): raise ValueError('目标点邻域没有有效深度')
    return float(np.median(vals)*depth_scale)

def pixel_depth_to_camera(depth_image,u,v,intrinsics,depth_scale=0.001,radius=2):
    return pixel_to_camera(u,v,robust_depth_m(depth_image,u,v,depth_scale,radius),intrinsics)
