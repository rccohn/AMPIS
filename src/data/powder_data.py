import json
import pathlib


def retrieve_data(json_path='particles'):
    '''
    Retrieves data for analysis
    Args:
        json_path: path to via json file containing image paths, labels, and masks. Can also input 'particles' and
                   'satellites,' which will use preset paths for default particle and satellite masks, respectively.
    Returns:
    '''

    via_root = pathlib.Path('..', '..', 'data', 'raw', 'via_2.0.8')
    # if path is specified as 'particles' or 'satellites', use pre-defined path
    default_paths = {'particles':  pathlib.Path(via_root, 'via_powder_particle_masks.json'),
                     'satellites': pathlib.Path(via_root, 'via_satellite_masks.json')}
    # otherwise, use the path specified by the user
    path = default_paths.get(json_path, pathlib.Path(json_path))
    assert path.is_file()  # verify final path exists

    # read data
    with open(path, 'rb') as f:
        data = json.load(f)

    img_root = pathlib.Path(data['_via_settings']['core']['default_filepath'])  # image paths are relative to here
    keys = list(data['_via_img_metadata'].keys())


    return img_root






def main():
    # default paths
    for path in ['particles', 'satellites']:
        p = retrieve_data(path)
        print(p)





if __name__ == '__main__':
    main()