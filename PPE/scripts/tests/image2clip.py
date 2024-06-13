import imageio.v3 as iio
import glob
import os

def image2clip(files):
    for filename in files:
        img = iio.imread(filename)
        iio.imwrite(filename.replace('.png', '.clip'), img)
        os.remove(filename)


def main():
    files = glob.glob('velocity_vec_vel_*.png')
    images = []
    for filename in files:
        img = iio.imread(filename)
        images.append(img)
    
    iio.imwrite('velocity_vec_vel.gif', images, duration = 500, loop = 0)

if __name__ == '__main__':
    main()

    # for filename in glob.glob('*.png'):
    #     img = iio.imread(filename)
    #     iio.imwrite(filename.replace('.png', '.clip'), img)
    #     os.remove(filename)