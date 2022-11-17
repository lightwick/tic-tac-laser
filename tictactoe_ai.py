import tensorflow as tf
import numpy as np

model = None

def init():
    global model
    model = tf.keras.models.load_model("output/after_gamma/92_5%124_1_9.h5")

def preprocess_grid(grid):
    processed = np.zeros([3,3])
    mapping = {' ':-1, 'O':0, 'X':1}
    for i in range(3):
        for j in range(3):
            processed[i][j] = mapping[grid[i][j]]
    
    state = tf.convert_to_tensor(processed)
    state = tf.expand_dims(state, 0)

    return state

def get_action_value(grid):
    grid = preprocess_grid(grid)
    return model(grid)


def get_action(grid):
    state = preprocess_grid(grid)
    action_value = model(state)

    action = None

    sorted_value = [[float(_x), _y] for _x,_y in zip(action_value[0], range(9))]
    sorted_value.sort(reverse=True)

    for i in sorted_value:
        action = i[1]
        if grid[action//3][action%3]==' ':
            break
    
    return action//3, action%3

if __name__=="__main__":
    init()
    grid=[['X','O',' '] for i in range(3)]
    print(get_action(grid))
else:
    init()