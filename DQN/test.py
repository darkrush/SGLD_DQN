from atari_wrappers import wrap_deepmind, make_atari
import numpy as np




env = make_atari('PongNoFrameskip-v4')
deep_env = wrap_deepmind(env, frame_stack = True)
deep_env.reset()
for i in range(10):
    obs ,reward, done,info = deep_env.step(0)
    print(obs)
    arr = np.swapaxes(np.array(obs), 2, 0)
    print(arr.dtype)
    