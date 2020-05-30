import cv2
from os import scandir, getcwd
from os.path import abspath,isfile

def ls_name(ruta = getcwd()):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

def ls(ruta = getcwd()):
    return [abspath(arch.path)[:-4] for arch in scandir(ruta) if arch.is_file() and abspath(arch.path).endswith("jpg")]


lista_nombres = ls('/home/artcs/Downloads/test-20200324T172331Z-001/test/RGB')

for ind,path in enumerate(lista_nombres):
    print(ind)
    print(path)
    path_destino = '/home/artcs/Desktop/Spherical-Convolution-master/SphereImages/test/'+str(ind)+'.jpg'
    print(path_destino)
    image = cv2.imread(path+'.jpg')
    print(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path_destino,gray)

