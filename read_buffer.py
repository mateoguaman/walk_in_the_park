import pickle

def main():
    saved_buffer_location = "/home/mateo/projects/walk_in_the_park/saved/buffers/buffer_3001"
    with open(saved_buffer_location, 'rb') as f:
        replay_buffer = pickle.load(f)
    import ipdb;ipdb.set_trace()

    loaded_data = replay_buffer.sample(1)

if __name__=="__main__":
    main()