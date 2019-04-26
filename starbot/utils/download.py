import os


def download(url, filename):
    rv = os.system('wget -O"{}" "{}"'.format(filename, url))
    if rv != 0:
        raise Exception("Downlaod {} failed".format(url))
