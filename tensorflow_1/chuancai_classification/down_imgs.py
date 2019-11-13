import os
import re
import sys
import cv2
import glob
import random
import shutil
import urllib
import argparse
import requests
from tqdm import tqdm
from multiprocessing import pool, Pool

relative_path = r'IMGS'

str_table = {
    '_z2c$q': ':',
    '_z&e3B': '.',
    'AzdH3F': '/'
}

char_table = {
    'w': 'a',
    'k': 'b',
    'v': 'c',
    '1': 'd',
    'j': 'e',
    'u': 'f',
    '2': 'g',
    'i': 'h',
    't': 'i',
    '3': 'j',
    'h': 'k',
    's': 'l',
    '4': 'm',
    'g': 'n',
    '5': 'o',
    'r': 'p',
    'q': 'q',
    '6': 'r',
    'f': 's',
    'p': 't',
    '7': 'u',
    'e': 'v',
    'o': 'w',
    '8': '1',
    'd': '2',
    'n': '3',
    '9': '4',
    'c': '5',
    'm': '6',
    '0': '7',
    'b': '8',
    'l': '9',
    'a': '0',
}

char_table = {ord(key): ord(value) for key, value in char_table.items()}


def decode(url):
    for key, value in str_table.items():
        url = url.replace(key, value)
    return url.translate(char_table)

def buildUrls(word, max_num):
    word = urllib.parse.quote(word)
    url = r"https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=result&fr=&sf=1&fmq=1573633891970_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E5%A4%AB%E5%A6%BB%E8%82%BA%E7%89%87"
    urls = (url.format(word=word, pn = str(x)) for x in range(0, max_num, 60))
    return urls


re_url = re.compile(r' "objURL" : "(.*?)"')


def resolveImgUrl(html):
    imgUrls = [decode(x) for x in re_url.findall((html))]
    return imgUrls


def downImgs(imgUrl, dirpath, imgName, imgTyoe):
    filename = os.path.join(dirpath, imgName)
    try:
        res = requests.get(imgUrl, timeout = 15)
        if str(res.status_code)[0] == '4':
            print(str(res.status_code), ":", imgUrl)
            return False
    except Exception as e:
        print(f'抛出异常：{imgUrl}')
        print(e)
        return False
    with open (filename + '.' + imgTyoe, 'wb') as f:
        f.write(res.content)
    return True


def downword(word):
    word_dir = os.path.join(sys.path[0], relative_path, word)
    os.makedirs(word_dir, exist_ok=True)

    index = 0
    numIMGS = 1000#Max images to be downloaded per keyword

    urls = buildUrls(word, numIMGS)
    for url in urls:
        html = requests.get(url, timeout = 10).content.decode('utf-8')
        imgUrls = resolveImgUrl(html)
        for url in imgUrls:
            try:
                if downImgs(url, word_dir, str(index + 1), 'jpg'):
                    index +=1
                    print(f"{word}:\t{index}张")
                if index == numIMGS:break
            except BaseException as e:
                print(url , e)
                continue


def del_bad(root):

    imgs = glob.glob(os.path.join(root, '*/*.jpg'), recursive=True)

    for img in tqdm(imgs):
        im = cv2.imread(img)
        if im is None:
            print(img)
            os.remove(img)
            continue
        h,w,_ = im.shape
        if h>w :    #top , bottom, left, right
            sq = cv2.copyMakeBorder(im, 0, 0, (h-w)//2, h-w-(h-w)//2,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        elif h < w:
            sq = cv2.copyMakeBorder(im, (w-h)//2, w-h-(w-h)//2, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        im = cv2.resize(im, (600, 600), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(img, im)


def split(root):
    subdirs = os.listdir(root)
    random.shuffle(subdirs)

    for subdir in subdirs:
        sub_imgs = os.listdir(os.path.join(root, subdir))
        N = len(sub_imgs)*9//10
        train = os.path.join(root, 'train', subdir)
        test = os.path.join(root, 'test', subdir)
        os.makedirs(train, mode=0o777, exist_ok=True)
        os.makedirs(test, mode=0o777, exist_ok=True)

        for sub_img in tqdm(sub_imgs[:N]):
            shutil.move(os.path.join(root, subdir, sub_img), train)
        for sub_img in tqdm(sub_imgs[N:]):
            shutil.move(os.path.join(root, subdir, sub_img), test)
        os.rmdir(os.path.join(root, subdir))


if __name__ == '__main__':
    with open(sys.argv[1], 'r', encoding='UTF-8') as f:
        words = (line.strip() for line in f.readlines())

    print(f'Keyword:{words}')

    with Pool() as pool:
        with tqdm(desc='Downloading Images') as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(downword, words))):
                pbar.update()
    del_bad(relative_path)
    split(relative_path)
