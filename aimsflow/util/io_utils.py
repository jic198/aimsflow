import os
import re
import sys
import bz2
import gzip
import shutil
import tarfile
import warnings
from glob import glob
from collections import OrderedDict, defaultdict

try:
    import yaml

    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper
except ImportError:
    yaml = None

PY_VERSION = sys.version_info


def get_str_btw_lines(in_str, start_str, end_str):
    target_str = re.findall(r"^.+?{}.+?\n((?:.*\n)+?).+?{}".format(start_str, end_str),
                            in_str, re.M)
    return target_str


def clean_lines(line_list, remove_newline=True, remove_comment=True):
    for i in line_list:
        clean_line = i
        if '#' in clean_line and remove_comment:
            index = clean_line.index('#')
            clean_line = clean_line[:index]
        clean_line = clean_line.strip()
        if (not remove_newline) or clean_line != '':
            yield clean_line


def zopen(filename, *args, **kwargs):
    file_ext = filename.split('.')[-1].upper()
    if file_ext == "BZ2":
        if PY_VERSION[0] >= 3:
            return bz2.open(filename, *args, **kwargs)
        else:
            args = list(args)
            if len(args) > 0:
                args[0] = ''.join([c for c in args[0] if c != 't'])
            if "mode" in kwargs:
                kwargs["mode"] = ''.join([c for c in kwargs["mode"]
                                          if c != 't'])
            return bz2.BZ2File(filename, *args, **kwargs)
    elif file_ext in ("GZ", 'Z'):
        return gzip.open(filename, *args, **kwargs)
    else:
        return open(filename, *args, **kwargs)


def file_to_str(filename):
    with zopen(filename) as f:
        return f.read().rstrip()


def str_to_file(string, filename):
    with zopen(filename, 'wt') as f:
        f.write(string)


def file_to_lines(filename, no_emptyline=False):
    outs = []
    with zopen(filename) as f:
        for i in f.readlines():
            if type(i) == bytes:
                i = i.decode()
            if no_emptyline and i == '\n':
                continue
            outs.append(i.strip("\n"))
        return outs


def immed_subdir(work_dir):
    return sorted([name for name in os.listdir(work_dir)
                   if os.path.isdir(os.path.join(work_dir, name))])


def immed_subdir_paths(work_dir):
    return sorted([os.path.join(work_dir, name) for name in os.listdir(work_dir)
                   if os.path.isdir(os.path.join(work_dir, name))])


def immed_files(work_dir):
    return sorted([name for name in os.listdir(work_dir)
                   if not os.path.isdir(os.path.join(work_dir, name))])


def immed_file_paths(folder):
    return sorted([os.path.join(folder, name) for name in os.listdir(folder)
                   if not os.path.isdir(os.path.join(folder, name))])


def make_path(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except FileNotFoundError:
            make_path(path.rsplit('/', 1)[0])
            make_path(path)


def loadfn(fn, *args, **kwargs):
    with zopen(fn) as fp:
        if "yaml" in fn.lower():
            if yaml is None:
                raise RuntimeError("Loading of YAML files is not "
                                   "possible as PyYAML is not installed.")
            if "Loader" not in kwargs:
                kwargs["Loader"] = Loader
            return yaml.load(fp, *args, **kwargs)


def ordered_loadfn(fn, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader): pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return loadfn(fn, Loader=OrderedLoader)


class Literal(str): pass


def literal_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:%s'
                                   % type(data).__name__, data)


def dict_presenter(dumper, data):
    return dumper.represent_dict(data.items())


def literal_dumpfn(obj, fn, *args, **kwargs):

    def process_literal(obj):
        for key, value in obj.items():
            try:
                process_literal(value)
            except AttributeError:
                try:
                    if "\n" in value:
                        obj[key] = Literal(value)
                except TypeError:
                    pass
        return obj

    obj = process_literal(obj)
    return dumpfn(obj, fn, *args, **kwargs)


def dumpfn(obj, fn, *args, **kwargs):
    with zopen(fn, "wt") as fp:
        if "yaml" in fn.lower():
            if yaml is None:
                raise RuntimeError("Loading of YAML files is not "
                                   "possible as PyYaml is not installed.")
            if "Dumper" not in kwargs:
                kwargs["Dumper"] = Dumper
            yaml.add_representer(Literal, literal_presenter)
            yaml.add_representer(OrderedDict, dict_presenter)
            yaml.add_representer(defaultdict, dict_presenter)
            yaml.add_representer(type(obj), dict_presenter)
            yaml.dump(obj, fp, *args, **kwargs)


def zpath(filename):
    for ext in ["", ".gz", ".GZ", ".bz2", ".BZ2", ".z", ".Z"]:
        zfilename = "{}{}".format(filename, ext)
        if os.path.exists(zfilename):
            return zfilename
    return filename


def copy_r(src, dst):
    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)
    make_path(abs_dst)
    for f in os.listdir(abs_src):
        f_path = os.path.join(abs_src, f)
        if os.path.isfile(f_path):
            shutil.copy(f_path, abs_dst)
        elif not abs_dst.startswith(f_path):
            copy_r(f_path, os.path.join(abs_dst, f))
        else:
            warnings.warn("Cannot copy %s to itself" % f_path)


def backup(filenames, prefix="error"):
    """
    Copied from Custodian
    Backup files to a tar.gz file. Used, for example, in backing up the
    files of an errored run before performing corrections.

    Args:
        filenames ([str]): List of files to backup. Supports wildcards, e.g.,
            *.*.
        prefix (str): prefix to the files. Defaults to error, which means a
            series of error.1.tar.gz, error.2.tar.gz, ... will be generated.
    """
    num = max([0] + [int(f.split(".")[1])
                     for f in glob("{}.*.tar.gz".format(prefix))])
    filename = "{}.{}.tar.gz".format(prefix, num + 1)
    with tarfile.open(filename, "w:gz") as tar:
        for fname in filenames:
            for f in glob(fname):
                tar.add(f)