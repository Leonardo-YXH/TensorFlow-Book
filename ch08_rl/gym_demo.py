import gym

def fuc1():
    env=gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space())

fuc1()