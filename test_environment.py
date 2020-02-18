import sys

def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    required_major = 3
    required_minor = 6
    print('checking python version major == 3 and minor version  == 6')
    print(sys.version_info)
    if system_major != required_major or system_minor != required_minor:
        raise('incorrect version. Please install python 3.6')

    print('importing packages')
    import torch
    import torchvision
    import pycocotools
    import detectron2
    from detectron2 import model_zoo
    print('successfully imported torch, torchvision, pycocotools, detectron2, detectron2.model_zoo')
    print('detecting gpu')
    print('cuda gpu detected: {} gpu name: {}'.format(torch.cuda.is_available(), torch.cuda.get_device_name()))
  
   
    
    if torch.cuda.is_available():
        print(">>> Development environment passes all tests!")
    else:
        print('all tests pass but no gpu detected')
    
    

if __name__ == '__main__':
    main()
