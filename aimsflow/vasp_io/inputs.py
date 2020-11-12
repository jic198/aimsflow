from __future__ import division

import re
import os
import sys
import math
import spglib
import warnings
import itertools
import collections
import numpy as np
from enum import Enum
from numpy.linalg import det
from tabulate import tabulate
from typing import Dict, Any, Optional, List

from aimsflow.symmetry.bandstructure import HighSymmKpath
from aimsflow.util import clean_lines, zopen, file_to_str, str_to_file, zpath, \
    loadfn, time_to_second, second_to_time, str_delimited, format_float
from aimsflow import Structure, Element, Lattice, PSP_DIR, MANAGER, TIME_TAG, \
    BATCH, WALLTIME

VASP_CONFIG = loadfn(os.path.join(os.path.dirname(__file__), "af_vasp.yaml"))
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_string(s):
    return "{}".format(s.strip())


def parse_bool(s):
    m = re.search("^\.?([T|F|t|f])[A-Za-z]*\.?", s)
    if m:
        if m.group(1) == "T" or m.group(1) == "t":
            return True
        else:
            return False
    raise ValueError(s + " should be a boolean type!")


def parse_float(s):
    return float(re.search("^-?\d*\.?\d*[e|E]?-?\d*", s).group(0))


def parse_int(s):
    return int(re.search("^-?[0-9]+", s).group(0))


def parse_list(s):
    return [float(y) for y in re.split("\s+", s.strip()) if not y.isalpha()]


class Poscar(object):
    def __init__(self, structure, comment=None, selective_dynamics=None,
                 true_names=True, velocities=None, predictor_corrector=None,
                 predictor_corrector_preamble=None):
        if structure.is_ordered:
            site_properties = {}
            if selective_dynamics:
                site_properties["selective_dynamics"] = selective_dynamics
            if velocities:
                site_properties["velocities"] = velocities
            if predictor_corrector:
                site_properties['predictor_corrector'] = predictor_corrector
            self.structure = structure.copy(site_properties=site_properties)
            self.true_names = true_names
            self.comment = structure.formula if comment is None else comment
            self.predictor_corrector_preamble = predictor_corrector_preamble
        else:
            raise ValueError("Structure with partial occupancies cannot be "
                             "convered into POSCAR!")

    def __repr__(self):
        return self.get_string()

    def __str__(self):
        return self.get_string()

    @property
    def site_symbols(self):
        syms = [site.specie.symbol for site in self.structure]
        return [a[0] for a in itertools.groupby(syms)]

    @property
    def num_sites(self):
        return sum(self.natoms)

    @property
    def natoms(self):
        syms = [site.specie.symbol for site in self.structure]
        return [len(tuple(a[1])) for a in itertools.groupby(syms)]

    @property
    def selective_dynamics(self):
        return self.structure.site_properties.get("selective_dynamics")

    @property
    def velocities(self):
        return self.structure.site_properties.get('velocities')

    @property
    def predictor_corrector(self):
        return self.structure.site_properties.get('predictor_corrector')

    @staticmethod
    def from_file(filename, read_velocities=True):
        return Poscar.from_string(file_to_str(filename), read_velocities=read_velocities)

    @staticmethod
    def from_string(data, default_symbol=None, read_velocities=True):
        chunks = re.split("\n\s*\n", data.rstrip())
        try:
            if chunks[0] == '':
                chunks.pop(0)
        except IndexError:
            raise ValueError("Empty POSCAR")
        lines = tuple(clean_lines(chunks[0].split('\n'), False))
        comment = lines[0]
        scale = float(lines[1])
        lattice = np.asarray([[float(j) for j in i.split()] for i in lines[2:5]])
        if scale < 0:
            vol = abs(det(lattice))
            lattice *= (-scale / vol) ** (1.0 / 3)
        else:
            lattice *= scale
        vasp5_symbols = False
        try:
            atom_nums = [int(i) for i in lines[5].split()]
            line_cursor = 6
        except ValueError:
            vasp5_symbols = True
            symbols = lines[5].split()
            atom_nums = [int(i) for i in lines[6].split()]
            atomic_symbols = []
            for i in range(len(atom_nums)):
                atomic_symbols.extend([symbols[i]] * atom_nums[i])
            line_cursor = 7

        poscar_type = lines[line_cursor].split()[0]
        sdynamics = False
        if poscar_type[0] in 'sS':
            sdynamics = True
            line_cursor += 1
            poscar_type = lines[line_cursor].split()[0]

        cart = poscar_type[0] in 'cCkK'
        num_sites = sum(atom_nums)

        if default_symbol:
            try:
                atomic_symbols = []
                for i in range(len(atom_nums)):
                    atomic_symbols.extend([default_symbol[i]] * atom_nums[i])
                vasp5_symbols = True
            except IndexError:
                pass

        if not vasp5_symbols:
            index = 3 if not sdynamics else 6
            try:
                atomic_symbols = [line.split()[index]
                                  for line in lines[line_cursor + 1: line_cursor + 1 + num_sites]]
                if not all([Element.is_valid_symbol(symbol)
                            for symbol in atomic_symbols]):
                    raise ValueError("Invalid symbol detected.")
                vasp5_symbols = True
            except (ValueError, IndexError):
                atomic_symbols = []
                for i in range(len(atom_nums)):
                    symbol = Element.from_z(i + 1).symbol
                    atomic_symbols.extend([symbol] * atom_nums[i])
                warnings.warn("Atomic symbol in POSCAR cannot be determined.\n"
                              "Defaulting to fake names {}.".format(" ".join(atomic_symbols)))
        coords = []
        select_dynamics = [] if sdynamics else None
        for i in range(num_sites):
            elements = lines[line_cursor + 1 + i].split()
            site_scale = scale if cart else 1
            coords.append([float(j) * site_scale for j in elements[:3]])
            if sdynamics:
                select_dynamics.append([j.upper()[0] == 'T'
                                        for j in elements[3:6]])
        struct = Structure(lattice, atomic_symbols, coords,
                           to_unit_cell=False, validate_proximity=False,
                           coords_cartesian=cart)

        if read_velocities:
            velocities = []
            if len(chunks) > 1:
                for line in chunks[1].strip().split('\n'):
                    velocities.append([float(i) for i in line.split()])

            predictor_corrector = []
            predictor_corrector_preamble = None
            # First line in chunk is a key in CONTCAR
            # Second line is POTIM
            # Third line is the thermostat parameters
            if len(chunks) > 2:
                lines = chunks[2].strip().split('\n')
                predictor_corrector_preamble = (lines[0] + '\n' + lines[1] +
                                                '\n' + lines[2])
                lines = lines[3:]
                for i in range(num_sites):
                    d1 = [float(j) for j in lines[i].split()]
                    d2 = [float(j) for j in lines[i + num_sites].split()]
                    d3 = [float(j) for j in lines[i + 2 * num_sites].split()]
                    predictor_corrector.append([d1, d2, d3])
        else:
            velocities = None
            predictor_corrector = None
            predictor_corrector_preamble = None

        return Poscar(struct, comment, select_dynamics, vasp5_symbols, velocities,
                      predictor_corrector, predictor_corrector_preamble)

    def get_string(self, direct=True, vasp4_compatible=False,
                   significant_figures=6):
        latt = self.structure.lattice
        if np.linalg.det(latt.matrix) < 0:
            latt = Lattice(-latt.matrix)

        lines = [self.comment, "1.0", str(latt)]
        if self.true_names and not vasp4_compatible:
            lines.append(" ".join(self.site_symbols))
        lines.append(" ".join([str(i) for i in self.natoms]))
        if self.selective_dynamics:
            lines.append("Selective dynamics")
        coord_type = "direct" if direct else "cartesian"
        lines.append("{}({})".format(coord_type, self.num_sites))

        format_str = "{{:.{0}f}}".format(significant_figures)
        for i, site in enumerate(self.structure):
            coords = site.frac_coords if direct else site.coords
            line = " ".join([format_str.format(c) for c in coords])
            if self.selective_dynamics is not None:
                sd = ["T" if j else "F" for j in self.selective_dynamics[i]]
                line += 3 * " %s" % tuple(sd)
            line += " " + site.species_string
            lines.append(line)

        if self.velocities:
            try:
                lines.append('')
                for v in self.velocities:
                    lines.append(' '.join([format_str.format(i) for i in v]))
            except:
                warnings.warn('Velocities are missing or corrupted.')

        if self.predictor_corrector:
            lines.append('')
            if self.predictor_corrector_preamble:
                lines.append(self.predictor_corrector_preamble)
                pred = np.array(self.predictor_corrector)
                for col in range(3):
                    for z in pred[:, col]:
                        lines.append(' '.join([format_str.format(i) for i in z]))
            else:
                warnings.warn('Preamble information missing or  corrupt. '
                              'Writing Poscar with no predictor corrector data.')

        return "\n".join(lines) + "\n"

    def write_file(self, filename, **kwargs):
        with zopen(filename, "wt") as f:
            f.write(self.get_string(**kwargs))


class BatchFile(dict):
    """
    BatchFile object for reading and writing a batch script. Essentially is a
    dictionary with batch tags
    """
    def __init__(self, params=None):
        """
        Create BatchFile object
        :param params: (dict) A set of batch tags
        """
        super(BatchFile, self).__init__()
        if params:
            self.update(params)
            self.manager = "PBS" if params.get("walltime") else "SLURM"
        self.queue_limit = {"hotel": "168",
                            "condo": "8",
                            "comet": "48"
                            }
        self.queue_ppn = {"hotel": "16:sandy",
                          "glean": "16:sandy",
                          "home": "28:broadwell",
                          "condo": "24:haswell"
                          }

    def __str__(self):
        return self.get_string()

    @staticmethod
    def from_file(filename):
        return BatchFile.from_string(file_to_str(filename))

    @staticmethod
    def from_string(string):
        lines = list(clean_lines(string.splitlines(), remove_comment=False))
        params = collections.OrderedDict()
        manager = "PBS" if "#PBS" in string else "SLURM"
        command = ""
        others = ""
        others_str = ["module", "ibrun", "mpirun", 'srun']
        for line in lines:
            m = re.match("#(?:PBS|SBATCH)\s+(\-\-*\w+\-*\w*\-*\w*)\=*\s*(.*)", line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                if "nodes" in val:
                    key = "nodes" if manager == "PBS" else "--nodes"
                    val = val.split("nodes=")[-1]
                if "walltime" in val:
                    key = "walltime"
                    val = val.split("walltime=")[-1]
                params[key] = val
            else:
                if "#!" in line:
                    params["header"] = line
                elif any([i in line for i in others_str]):
                    others += line + "\n"
                else:
                    command += line + "\n"
        try:
            exe = re.search("\s(\{*\w+\}*)\s*>\s*vasp.out", others).group(1)
        except AttributeError:
            exe = None
        params.update({"command": command, "others": others, "exe": exe})
        return BatchFile(params)

    @staticmethod
    def convert_batch(old_batch):
        if MANAGER == old_batch.manager:
            sys.stderr.write("The batch is already in %s format\n" % MANAGER)
            return old_batch

        new_batch = BatchFile.from_string(BATCH)
        if MANAGER == "SLURM":
            walltime = time_to_second(old_batch["walltime"])
            if walltime > new_batch.time_limit("comet"):
                walltime = new_batch.time_limit("comet")
            new_batch.update({"--mail-user": old_batch.get("-M"),
                              "-J": old_batch.get("-N"),
                              "-t": second_to_time(walltime)})
        else:
            walltime = time_to_second(old_batch["-t"])
            if walltime > new_batch.time_limit("condo"):
                new_batch.change_queue("home")
            new_batch.update({"-M": old_batch.get("--mail-user"),
                              "-N": old_batch.get("-J"),
                              "walltime": old_batch.get("-t")})

        if "aimsflow" in old_batch["command"]:
            command = [i for i in new_batch["command"].split("\n")
                       if i.startswith("cd")]
            command.append(re.findall("(aimsflow.*\n)", old_batch["command"])[0])
            new_batch.update({"command": "\n".join(command), "others": ""})
            err = new_batch["-e"]
            out = new_batch["-o"]
            new_batch.update({"-e": err.replace("err.", "err.check."),
                              "-o": out.replace("out.", "out.check.")})
        else:
            command = "".join(re.findall("(cp.*\n)", old_batch["command"]))
            if "{command}" not in old_batch["command"]:
                new_batch["command"] = new_batch["command"].format(command=command)
        if old_batch["exe"] is not None:
            new_batch["others"] = re.sub("\{*\w+\}*(?=\s*>\s*vasp.out)",
                                         old_batch["exe"], new_batch["others"])
        return BatchFile(new_batch)

    def time_limit(self, queue):
        return time_to_second(self.queue_limit[queue])

    def get_string(self):
        keys = list(self)
        [keys.remove(i) for i in ["header", "others", "command", "exe"]]
        lines = []
        if self.manager == "PBS":
            for k in keys:
                if k in ["walltime", "nodes"]:
                    lines.append(["#PBS -l {}=".format(k), self[k]])
                else:
                    lines.append(["#PBS " + k, self[k]])
            lines.append([self["command"] + self["others"]])
            string = str_delimited(lines, header=[self["header"]], delimiter=" ")
            string = re.sub("nodes=\s+", "nodes=", string)
            return re.sub("walltime=\s+", "walltime=", string)
        else:
            for k in keys:
                lines.append(["#SBATCH " + k, self[k]])
            lines.append([self["command"] + self["others"]])
            return str_delimited(lines, header=[self["header"]], delimiter=" ")

    def write_file(self, filename):
        str_to_file(self.__str__(), filename)

    def change_queue(self, queue):
        if MANAGER == "SLURM":
            raise IOError("In SBATCH system, there is no changing "
                          "queue option.\n")

        ppn = self.queue_ppn[queue]
        self.update({"-q": queue})
        ppn = re.sub("ppn=.*", "ppn={}".format(ppn),
                     self.get("nodes", "1:ppn=16"))
        self.update({"nodes": ppn})
        if queue in self.queue_limit:
            walltime = self[TIME_TAG]
            if time_to_second(walltime) > self.time_limit(queue):
                queue_limit = self.queue_limit[queue]
                sys.stderr.write("Maximum walltime for %s is %s hrs. "
                                 "The current %s will be changed to %s hrs.\n"
                                 % (queue, queue_limit, walltime, queue_limit))
                self.update({TIME_TAG: "%s:00:00" % queue_limit})

    def change_walltime(self, walltime):
        seconds = time_to_second(walltime)
        if MANAGER == "PBS":
            queue = self["-q"]
            if queue in ["condo", "hotel"]:
                if seconds > self.time_limit(queue):
                    self.change_queue("home")
        self.update({TIME_TAG: second_to_time(seconds)})

    def change_processors(self, ppn):
        if MANAGER == "SLURM":
            self.update({"--ntasks-per-node": ppn})
        else:
            if ppn == 16:
                ppn = "{}:sandy".format(ppn)
            elif ppn == 24:
                ppn = "{}:haswell".format(ppn)
            elif ppn == 28:
                ppn = "{}:broadwell".format(ppn)
            ppn = re.sub("ppn=.*", "ppn={}".format(ppn),
                         self.get("nodes", "1:ppn=16"))
            self.update({"nodes": ppn})

    def change_jobname(self, jobname):
        if MANAGER == "SLURM":
            self.update({"-J": jobname})
        else:
            self.update({"-N": jobname})

    def change_mail_type(self, name):
        if MANAGER == "SLURM":
            m_type = ['BEGIN', 'END', 'FAIL', 'REQUEUE', 'STAGE_OUT', 'NONE',
                      'ALL', 'STAGE_OUT', 'TIME_LIMIT', 'TIME_LIMIT_90',
                      'TIME_LIMIT_80', 'TIME_LIMIT_50', 'ARRAY_TASKS']
            n_list = name.split(",")
            if all([i in m_type for i in n_list]):
                self.update({"--mail-type": name})
            else:
                raise IOError("%s is not supported in SLURM" % name)
        else:
            if all([i in ['a', 'b', 'e'] for i in name]):
                self.update({"-m": name})
            else:
                raise IOError("%s is not supported in PBS" % name)

    def change_exe(self, exe):
        self["others"] = re.sub("\{*\w+\}*(?=\s*>\s*vasp.out)",
                                exe, self["others"])


class Incar(dict):
    """
    Incar object for reading and writing a INCAR file. Essentially is a
    dictionary with VASP tags
    """
    def __init__(self, params=None):
        """
        Create an Incar object
        :param params: (dict) A set of VASP tags
        """
        super(Incar, self).__init__()
        if params:
            self.update(params)

    def __str__(self):
        return self.get_string(sort_keys=True, pretty=False)

    @staticmethod
    def from_file(filename):
        return Incar.from_string(file_to_str(filename))

    @staticmethod
    def from_string(string):
        lines = list(clean_lines(string.splitlines(), remove_comment=False))
        params = {}
        for line in lines:
            m = re.match("(\w+)\s*=\s*(.*)", line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                val = Incar.proc_val(key, val)
                params[key] = val
        return Incar(params)

    @staticmethod
    def proc_val(key, val):
        list_keys = ("LDAUU", "LDAUL", "LDAUJ", "MAGMOM", "DIPOL")
        bool_keys = ("LDAU", "LWAVE", "LSCALU", "LCHARG", "LPLANE",
                     "LHFCALC", "ADDGRID", "LSORBIT", "LNONCOLLINEAR")
        float_keys = ("EDIFF", "SIGMA", "TIME", "ENCUTFOCK", "HFSCREEN",
                      "POTIM", "EDIFFG")
        int_keys = ("NSW", "NBANDS", "NELMIN", "ISIF", "IBRION", "ISPIN",
                    "ICHARG", "NELM", "ISMEAR", "NPAR", "LDAUPRINT", "LMAXMIX",
                    "ENCUT", "NSIM", "NKRED", "NUPDOWN", "ISPIND", "LDAUTYPE")

        def smart_int_or_float(numstr):
            if numstr.find(".") != -1 or numstr.lower().find("e") != -1:
                return float(numstr)
            else:
                return int(numstr)

        try:
            if key in list_keys:
                output = []
                toks = re.findall(r"(-?\d+\.?\d*)\*?(-?\d+\.?\d*)?\*?(-?\d+\.?\d*)?", val)
                for tok in toks:
                    if tok[2] and "3" in tok[0]:
                        output.extend(
                            [smart_int_or_float(tok[2])] * int(tok[0]) * int(tok[1]))
                    elif tok[1]:
                        output.extend([smart_int_or_float(tok[1])] * int(tok[0]))
                    else:
                        output.append(smart_int_or_float(tok[0]))
                return output
            if key in bool_keys:
                m = re.match(r"^\.?([T|F|t|f])[A-Za-z]*\.?", val)
                if m:
                    if m.group(1) == "T" or m.group(1) == "t":
                        return True
                    else:
                        return False
                raise ValueError(key + " should be a boolean type!")
            if key in float_keys:
                return float(re.search(r"^-?\d*\.?\d*[e|E]?-?\d*", val).group(0))
            if key in int_keys:
                return int(re.match(r"^-?[0-9]+", val).group(0))
        except ValueError:
            pass

        try:
            val = int(val)
            return val
        except ValueError:
            pass

        try:
            val = float(val)
            return val
        except ValueError:
            pass

        if "true" in val.lower():
            return True

        if "false" in val.lower():
            return False

        try:
            if key not in ("TITEL", "SYSTEM"):
                return re.search(r"^-?[0-9]+", val.capitalize()).group(0)
            else:
                return val.capitalize()
        except:
            return val.capitalize()

    def get_string(self, sort_keys=False, pretty=False):
        keys = self.keys()
        if sort_keys:
            keys = sorted(keys)
        lines = []
        for k in keys:
            if k == "MAGMOM" and isinstance(self[k], list):
                value = []
                if isinstance(self[k][0], list) and (self.get("LSORBIT") or
                                                     self.get("LNONCOLLINEAR")):
                    self[k] = [format_float(i, no_one=False)
                               for i in np.matrix(self[k]).A1]
                for m, g in itertools.groupby(self[k]):
                    value.append("{}*{}".format(len(tuple(g)), m))
                lines.append([k, " ".join(value)])
            elif isinstance(self[k], list):
                lines.append([k, " ".join([str(i) for i in self[k]])])
            else:
                lines.append([k, self[k]])

        if pretty:
            return str(tabulate([[l[0], "=", l[1]] for l in lines],
                                tablefmt="plain"))
        else:
            return str_delimited(lines, None, " = ") + "\n"

    def write_file(self, filename):
        str_to_file(self.__str__(), filename)


class PotcarSingle(object):
    functional_tags = {"PE": "PBE", "91": "PW91", "CA": "LDA"}
    parse_tags = {"LULTRA": parse_bool,
                  "LCOR": parse_bool,
                  "LPAW": parse_bool,
                  "EATOM": parse_float,
                  "RPACOR": parse_float,
                  "POMASS": parse_float,
                  "ZVAL": parse_float,
                  "RCORE": parse_float,
                  "RWIGS": parse_float,
                  "ENMAX": parse_float,
                  "ENMIN": parse_float,
                  "EAUG": parse_float,
                  "DEXC": parse_float,
                  "RMAX": parse_float,
                  "RAUG": parse_float,
                  "RDEP": parse_float,
                  "RDEPT": parse_float,
                  "QCUT": parse_float,
                  "QGAM": parse_float,
                  "RCLOC": parse_float,
                  "IUNSCR": parse_int,
                  "ICORE": parse_int,
                  "NDATA": parse_int,
                  "VRHFIN": parse_string,
                  "LEXCH": parse_string,
                  "TITEL": parse_string,
                  "STEP": parse_list,
                  "RRKJ": parse_list,
                  "GGA": parse_list}

    def __init__(self, data):
        self.data = data
        self.header = data.split("\n")[0].strip()
        search_lines = re.search("(parameters from.*?PSCTR-controll parameters)",
                                 data, re.S).group(1)
        self.keywords = {}
        for key, val in re.findall("(\S+)\s*=\s*(.*?)(?=;|$)",
                                   search_lines, re.M):
            self.keywords[key] = self.parse_tags[key](val)
        try:
            self.symbol = self.keywords["TITEL"].split(" ")[1].strip()
        except IndexError:
            self.symbol = self.keywords["TITEL"].strip()
        self.functional = self.functional_tags[self.keywords["LEXCH"]]
        if self.functional == "PBE":
            if "mkinetic" in data:
                self.functional = "PBE_52"
        elif self.functional == "LDA":
            if "US" in self.keywords["TITEL"]:
                self.functional = "LDA_US"

    def __str__(self):
        return self.data + "\n"

    @staticmethod
    def from_file(filename):
        return PotcarSingle(file_to_str(filename).decode("utf-8"))


class Potcar(list):
    def __init__(self, psps, functional="PBE"):
        super(Potcar, self).__init__()
        self.functional = functional
        if isinstance(psps, list):
            self.extend(psps)

    def __str__(self):
        return "\n".join([str(potcar).strip("\n") for potcar in self]) + "\n"

    def write_file(self, filename="POTCAR"):
        str_to_file(self.__str__(), filename)

    @staticmethod
    def from_file(filename):
        return Potcar.from_string(file_to_str(filename))

    @staticmethod
    def from_string(string):
        chuncks = re.findall("\n?(\s*.*?End of Dataset)", string, re.S)
        psps = [PotcarSingle(i) for i in chuncks]
        all_funs = [i.functional for i in psps]
        functional = psps[0].functional if "PBE_52" not in all_funs else "PBE_52"
        return Potcar(psps, functional)

    @staticmethod
    def from_elements(elements, functional="PBE"):
        try:
            d = PSP_DIR
        except KeyError:
            raise KeyError("Please set the AF_VASP_PSP_DIR environment in "
                           ".afrc.yaml. E.g. aimsflow config -a "
                           "AF_VASP_PSP_DIR ~/psps")
        psps = []
        potcar_setting = VASP_CONFIG["POTCAR"]
        fundir = VASP_CONFIG["FUNDIR"][functional]
        for el in elements:
            psp = []
            symbol = potcar_setting.get(el, el)
            paths_to_try = [os.path.join(PSP_DIR, fundir, f"POTCAR.{symbol}"),
                            os.path.join(PSP_DIR, fundir, symbol, "POTCAR")]
            for p in paths_to_try:
                p = os.path.expanduser(p)
                p = zpath(p)
                if os.path.exists(p):
                    psp = PotcarSingle.from_file(p)
                    break
            if psp:
                psps.append(psp)
            else:
                raise IOError("Cannot find the POTCAR with functional %s and "
                              "label %s" % (functional, symbol))
        return Potcar(psps, functional)


class Kpoints_supported_modes(Enum):
    Automatic = 0
    Gamma = 1
    MonKhorst = 2
    Line_mode = 3
    Cartesian = 4
    Reciprocal = 5

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        c = s.lower()[0]
        for m in Kpoints_supported_modes:
            if m.name.lower()[0] == c:
                return m
        raise ValueError("Can't interprete Kpoint mode %s" % s)


class Kpoints(object):
    supported_modes = Kpoints_supported_modes

    def __init__(self, comment="Default gamma", num_kpts=0,
                 style=supported_modes.Gamma,
                 kpts=((1, 1, 1),), kpts_shift=(0, 0, 0),
                 kpts_weights=None, coord_type=None, labels=None,
                 tet_number=0, tet_weight=0, tet_connections=None):
        if num_kpts > 0 and (not labels) and (not kpts_weights):
            raise ValueError("For explicit or line-mode kpoints, either the "
                             "labels or kpts_weights must be specified.")
        self.comment = comment
        self.num_kpts = num_kpts
        self.style = style
        self.kpts = kpts
        self.kpts_shift = kpts_shift
        self.kpts_weights = kpts_weights
        self.coord_type = coord_type
        self.labels = labels
        self.tet_number = tet_number
        self.tet_weight = tet_weight
        self.tet_connections = tet_connections

    def __str__(self):
        lines = [self.comment, str(self.num_kpts), str(self.style)]
        style = self.style.name.lower()[0]
        if style == "l":
            lines.append(self.coord_type)
        for i in range(len(self.kpts)):
            lines.append(" ".join([str(j) for j in self.kpts[i]]))
            if style == "l":
                lines[-1] += " ! " + self.labels[i]
                if i % 2 == 1:
                    lines[-1] += "\n"
            elif self.num_kpts > 0:
                if self.labels is not None:
                    lines[-1] += " %i %s" % (self.kpts_weights[i],
                                             self.labels[i])
                else:
                    lines[-1] += " %i" % self.kpts_weights[i]
        # Print tetrahedron parameters if the number of tetrahedrons > 0
        if style not in "lagm" and self.tet_number > 0:
            lines.append("Tetrahedron")
            lines.append("%d %f" % (self.tet_number, self.tet_weight))
            for sym_weight, vertices in self.tet_connections:
                lines.append("%d %s" % (sym_weight, " ".join(vertices)))
        # Print shifts for automatic kpoints types if not zero
        if self.num_kpts <= 0 and tuple(self.kpts_shift) != (0, 0, 0):
            lines.append(" ".join("%.3f" % f for f in self.kpts_shift))
        return "\n".join(lines) + "\n"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def automatic(subdivisions):
        return Kpoints("Fully automatic kpoint scheme", 0,
                       style=Kpoints_supported_modes.Automatic,
                       kpts=[[subdivisions]])

    @staticmethod
    def gamma_automatic(kpts=(1, 1, 1), shift=(0, 0, 0)):
        return Kpoints("Automatic kpoint scheme", 0,
                       Kpoints.supported_modes.Gamma, kpts=[kpts],
                       kpts_shift=shift)

    @staticmethod
    def monkhorst_automatic(kpts=(2, 2, 2), shift=(0, 0, 0)):
        return Kpoints("Automatic kpoint scheme", 0,
                       Kpoints.supported_modes.MonKhorst, kpts=[kpts],
                       kpts_shift=shift)

    @staticmethod
    def automatic_density(structure, kpts, force_gamma=False):
        """
        Returns an automatic Kpoint object based on a structure and a kpoint
        density. Uses Gamma centered meshes for hexagonal cells and
        Monkhorst-Pack grids otherwise.

        Algorithm:
            Uses a simple approach scaling the number of divisions along each
            reciprocal lattice vector proportional to its length.
        :param structure:
        :param kpts:
        :param force_gamma:
        :return:
        """
        comment = "aimsflow generated KPOINTS with grid density " \
                  "= %.0f / atom" % kpts
        latt = structure.lattice
        lengths = latt.abc
        ngrid = kpts / structure.num_sites
        mult = (ngrid * lengths[0] * lengths[1] * lengths[2]) ** (1 / 3)
        num_div = [int(math.floor(max(mult / l, 1))) for l in lengths]
        is_hax = latt.is_hex()
        has_odd = any([i % 2 == 1 for i in num_div])

        if has_odd or is_hax or force_gamma:
            style = Kpoints.supported_modes.Gamma
        else:
            style = Kpoints.supported_modes.MonKhorst

        return Kpoints(comment, 0, style, [num_div], [0, 0, 0])

    @staticmethod
    def automatic_density_by_vol(structure, density, force_gamma=False):
        """
        Returns an automatic Kpoint object based on a structure and a kpoint
        density per inverse Angstrom of reciprocal cell.

        :param structure: Structure object
        :param density: Grid density per Angstrom^(-3) of reciprocal cell
        :param force_gamma: Force a gamma centered mesh
        :return: Kpoints object
        """
        vol = structure.lattice.reciprocal_lattice.volume
        kpts = density * vol * structure.num_sites
        return Kpoints.automatic_density(structure, kpts,
                                         force_gamma=force_gamma)

    @staticmethod
    def from_file(filename):
        return Kpoints.from_string(file_to_str(filename))

    @staticmethod
    def from_string(string):
        lines = [line.strip() for line in string.splitlines()]
        comment = lines[0]
        num_kpts = int(lines[1].split()[0])
        style = lines[2].lower()[0]
        # Fully automatic KPOINTS
        if style == "a":
            return Kpoints.automatic(int(lines[3]))
        # Automatic gamma and Monk KPOINTS, with optional shift
        if style in ["g", "m"]:
            kpts = [int(i) for i in lines[3].split()]
            kpts_shift = (0, 0, 0)
            if len(lines) > 4:
                try:
                    kpts_shift = [int(i) for i in lines[4].split()]
                except ValueError:
                    pass
            return Kpoints.gamma_automatic(kpts, kpts_shift) if style == "g" \
                else Kpoints.monkhorst_automatic(kpts, kpts_shift)
        # Automatic kpoints with basis
        if num_kpts <= 0:
            style = Kpoints.supported_modes.Cartesian if style in "ck" \
                else Kpoints.supported_modes.Reciprocal
            kpts = [[float(j) for j in lines[i].split()] for i in range(3, 6)]
            kpts_shift = [float(i) for i in lines[6].split()]
            return Kpoints(comment, num_kpts, style, kpts, kpts_shift)
        # Line-mode KPOINTS, usually used with band structure
        if style == "l":
            coord_type = "Catesian" if lines[3].lower()[0] in "ck" \
                else "Reciprocal"
            style = Kpoints_supported_modes.Line_mode
            patt = re.findall("([\d\.*\-]+)\s+([\d\.*\-]+)\s+([\d\.*\-]+)\s+"
                              "!\s*(.*)", "\n".join(lines[4:]))
            patt = np.array(patt)
            kpts = [[float(j) for j in i] for i in patt[:, :3]]
            labels = [i.strip() for i in patt[:, 3]]
            return Kpoints(comment, num_kpts, style, kpts,
                           coord_type=coord_type, labels=labels)
        # Assume explicit KPOINTS if all else fails
        style = Kpoints.supported_modes.Cartesian if style in "ck" \
            else Kpoints.supported_modes.Reciprocal
        kpts = []
        kpts_weights = []
        labels = []
        tet_number = 0
        tet_weight = 0
        tet_connections = None

        for i in range(3, 3 + num_kpts):
            toks = lines[i].split()
            kpts.append([float(j) for j in toks[:3]])
            kpts_weights.append(float(toks[3]))
            if len(toks) > 4:
                labels.append(toks[4])
            else:
                labels.append(None)
        try:
            # Deal with tetrahedron method
            lines = lines[3 + num_kpts:]
            if lines[0].strip().lower()[0] == "t":
                toks = lines[1].split()
                tet_number = int(toks[0])
                tet_weight = float(toks[1])
                tet_connections = []
                for i in range(2, 2 + tet_number):
                    toks = lines[i].split()
                    tet_connections.append((int(toks[0]), [int(j) for j in toks[1:]]))
        except IndexError:
            pass
        return Kpoints(comment, num_kpts, style, kpts,
                       kpts_weights=kpts_weights, labels=labels,
                       tet_number=tet_number, tet_weight=tet_weight,
                       tet_connections=tet_connections)

    def write_file(self, filename):
        str_to_file(self.__str__(), filename)


class Lobsterin(dict):
    """
    Janine George, Marco Esters
    This class can handle and generate lobsterin files
    Furthermore, it can also modify INCAR files for lobster, generate KPOINT files for fatband calculations in Lobster,
    and generate the standard primitive cells in a POSCAR file that are needed for the fatband calculations.
    There are also several standard lobsterin files that can be easily generated.
    """

    # all keywords known to this class so far
    # reminder: lobster is not case sensitive
    AVAILABLEKEYWORDS = ['COHPstartEnergy', 'COHPendEnergy', 'basisSet', 'cohpGenerator',
                         'gaussianSmearingWidth', 'saveProjectionToFile', 'basisfunctions', 'skipdos',
                         'skipcohp', 'skipcoop', 'skipPopulationAnalysis', 'skipGrossPopulation',
                         'userecommendedbasisfunctions', 'loadProjectionFromFile', 'forceEnergyRange',
                         'DensityOfEnergy', 'BWDF', 'BWDFCOHP', 'skipProjection', 'createFatband',
                         'writeBasisFunctions', 'writeMatricesToFile', 'realspaceHamiltonian',
                         'realspaceOverlap', 'printPAWRealSpaceWavefunction', 'printLCAORealSpaceWavefunction',
                         'noFFTforVisualization', 'RMSp', 'onlyReadVasprun.xml', 'noMemoryMappedFiles',
                         'skipPAWOrthonormalityTest', 'doNotIgnoreExcessiveBands', 'doNotUseAbsoluteSpilling',
                         'skipReOrthonormalization', 'forceV1HMatrix', 'useOriginalTetrahedronMethod',
                         'useDecimalPlaces', 'kSpaceCOHP']

    # keyword + one float can be used in file
    FLOATKEYWORDS = ['COHPstartEnergy', 'COHPendEnergy', 'gaussianSmearingWidth', 'useDecimalPlaces', 'COHPSteps']
    # one of these keywords +endstring can be used in file
    STRINGKEYWORDS = ['basisSet', 'cohpGenerator', 'realspaceHamiltonian', 'realspaceOverlap',
                      'printPAWRealSpaceWavefunction', 'printLCAORealSpaceWavefunction', 'kSpaceCOHP']
    # the keyword alone will turn on or off a function
    BOOLEANKEYWORDS = ['saveProjectionToFile', 'skipdos', 'skipcohp', 'skipcoop', 'loadProjectionFromFile',
                       'forceEnergyRange', 'DensityOfEnergy', 'BWDF', 'BWDFCOHP', 'skipPopulationAnalysis',
                       'skipGrossPopulation', 'userecommendedbasisfunctions', 'skipProjection',
                       'writeBasisFunctions', 'writeMatricesToFile', 'noFFTforVisualization', 'RMSp',
                       'onlyReadVasprun.xml', 'noMemoryMappedFiles', 'skipPAWOrthonormalityTest',
                       'doNotIgnoreExcessiveBands', 'doNotUseAbsoluteSpilling', 'skipReOrthonormalization',
                       'forceV1HMatrix', 'useOriginalTetrahedronMethod', 'forceEnergyRange', 'bandwiseSpilling',
                       'kpointwiseSpilling']
    # several of these keywords + ending can be used in a lobsterin file:
    LISTKEYWORDS = ['basisfunctions', 'cohpbetween', 'createFatband']

    def __init__(self, settingsdict: dict):
        """
        Args:
            settingsdict: dict to initialize Lobsterin
        """
        super().__init__()
        # check for duplicates
        listkey = [key.lower() for key in settingsdict.keys()]
        if len(listkey) != len(list(set(listkey))):
            raise IOError("There are duplicates for the keywords! The program will stop here.")
        self.update(settingsdict)

    def __setitem__(self, key, val):
        """
        Add parameter-val pair to Lobsterin.  Warns if parameter is not in list of
        valid lobsterintags. Also cleans the parameter and val by stripping
        leading and trailing white spaces. Similar to INCAR class.
        """
        # due to the missing case sensitivity of lobster, the following code is neccessary
        found = False
        for key_here in self.keys():
            if key.strip().lower() == key_here.lower():
                new_key = key_here
                found = True
        if not found:
            new_key = key
        if new_key.lower() not in [element.lower() for element in Lobsterin.AVAILABLEKEYWORDS]:
            raise ValueError("Key is currently not available")

        super().__setitem__(new_key, val.strip() if isinstance(val, str) else val)

    def __getitem__(self, item):
        """
        implements getitem from dict to avoid problems with cases
        """
        found = False
        for key_here in self.keys():
            if item.strip().lower() == key_here.lower():
                new_key = key_here
                found = True
        if not found:
            new_key = item

        val = dict.__getitem__(self, new_key)
        return val

    def diff(self, other):
        """
        Diff function for lobsterin. Compares two lobsterin and indicates which parameters are the same.
        Similar to the diff in INCAR.
        Args:
            other (Lobsterin): Lobsterin object to compare to
        Returns:
            dict with differences and similarities
        """
        similar_param = {}
        different_param = {}
        key_list_others = [element.lower() for element in other.keys()]

        for k1, v1 in self.items():
            k1lower = k1.lower()
            if k1lower not in key_list_others:
                different_param[k1.upper()] = {"lobsterin1": v1, "lobsterin2": None}
            else:
                for key_here in other.keys():
                    if k1.lower() == key_here.lower():
                        new_key = key_here

                if isinstance(v1, str):
                    if v1.strip().lower() != other[new_key].strip().lower():

                        different_param[k1.upper()] = {"lobsterin1": v1, "lobsterin2": other[new_key]}
                    else:
                        similar_param[k1.upper()] = v1
                elif isinstance(v1, list):
                    new_set1 = {element.strip().lower() for element in v1}
                    new_set2 = {element.strip().lower() for element in other[new_key]}
                    if new_set1 != new_set2:
                        different_param[k1.upper()] = {"lobsterin1": v1, "lobsterin2": other[new_key]}
                else:
                    if v1 != other[new_key]:
                        different_param[k1.upper()] = {"lobsterin1": v1, "lobsterin2": other[new_key]}
                    else:
                        similar_param[k1.upper()] = v1

        for k2, v2 in other.items():
            if k2.upper() not in similar_param and k2.upper() not in different_param:
                for key_here in self.keys():
                    if k2.lower() == key_here.lower():
                        new_key = key_here
                    else:
                        new_key = k2
                if new_key not in self:
                    different_param[k2.upper()] = {"lobsterin1": None, "lobsterin2": v2}
        return {"Same": similar_param, "Different": different_param}

    def _get_nbands(self, structure: Structure):
        """
        get number of nbands
        """
        if self.get("basisfunctions") is None:
            raise IOError("No basis functions are provided. The program cannot calculate nbands.")

        basis_functions = []  # type: List[str]
        for string_basis in self["basisfunctions"]:
            # string_basis.lstrip()
            string_basis_raw = string_basis.strip().split(" ")
            while "" in string_basis_raw:
                string_basis_raw.remove("")
            for i in range(0, int(structure.composition.get_el_amt_dict()[string_basis_raw[0]])):
                basis_functions.extend(string_basis_raw[1:])

        no_basis_functions = 0
        for basis in basis_functions:
            if "s" in basis:
                no_basis_functions = no_basis_functions + 1
            elif "p" in basis:
                no_basis_functions = no_basis_functions + 3
            elif "d" in basis:
                no_basis_functions = no_basis_functions + 5
            elif "f" in basis:
                no_basis_functions = no_basis_functions + 7

        return int(no_basis_functions)

    def write_lobsterin(self, path="lobsterin", overwritedict=None):
        """
        writes a lobsterin file
        Args:
            path (str): filename of the lobsterin file that will be written
            overwritedict (dict): dict that can be used to overwrite lobsterin, e.g. {"skipdos": True}
        """

        # will overwrite previous entries
        # has to search first if entry is already in Lobsterindict (due to case insensitivity)
        if overwritedict is not None:
            for key, entry in overwritedict.items():
                found = False
                for key2 in self.keys():
                    if key.lower() == key2.lower():
                        self[key2] = entry
                        found = True
                if not found:
                    self[key] = entry

        filename = path
        with open(filename, 'w') as f:
            for key in Lobsterin.AVAILABLEKEYWORDS:
                if key.lower() in [element.lower() for element in self.keys()]:
                    if key.lower() in [element.lower() for element in Lobsterin.FLOATKEYWORDS]:
                        f.write(key + ' ' + str(self.get(key)) + '\n')
                    elif key.lower() in [element.lower() for element in Lobsterin.BOOLEANKEYWORDS]:
                        # checks if entry is True or False
                        for key_here in self.keys():
                            if key.lower() == key_here.lower():
                                new_key = key_here
                        if self.get(new_key):
                            f.write(key + '\n')
                    elif key.lower() in [element.lower() for element in Lobsterin.STRINGKEYWORDS]:
                        f.write(key + ' ' + str(self.get(key) + '\n'))
                    elif key.lower() in [element.lower() for element in Lobsterin.LISTKEYWORDS]:
                        for entry in self.get(key):
                            f.write(key + ' ' + str(entry) + '\n')

    def as_dict(self):
        """
        :return: MSONable dict
        """
        d = dict(self)
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        return d

    @classmethod
    def from_dict(cls, d):
        """
        :param d: Dict representation
        :return: Lobsterin
        """
        return Lobsterin({k: v for k, v in d.items() if k not in ["@module",
                                                                  "@class"]})

    def write_INCAR(self, incar_input: str = "INCAR", incar_output: str = "INCAR.lobster",
                    poscar_input: str = "POSCAR", isym: int = -1,
                    further_settings: dict = None):
        """
        Will only make the run static, insert nbands, make ISYM=-1, set LWAVE=True and write a new INCAR.
        You have to check for the rest.
        Args:
            incar_input (str): path to input INCAR
            incar_output (str): path to output INCAR
            poscar_input (str): path to input POSCAR
            isym (int): isym equal to -1 or 0 are possible. Current Lobster version only allow -1.
            further_settings (dict): A dict can be used to include further settings, e.g. {"ISMEAR":-5}
        """
        # reads old incar from file, this one will be modified
        incar = Incar.from_file(incar_input)
        warnings.warn("Please check your incar_input before using it. This method only changes three settings!")
        if isym == -1:
            incar["ISYM"] = -1
        elif isym == 0:
            incar["ISYM"] = 0
        else:
            ValueError("isym has to be -1 or 0.")
        incar["NSW"] = 0
        incar["LWAVE"] = True
        # get nbands from _get_nbands (use basis set that is inserted)
        incar["NBANDS"] = self._get_nbands(Structure.from_file(poscar_input))
        if further_settings is not None:
            for key, item in further_settings.items():
                incar[key] = further_settings[key]
        # print it to file
        incar.write_file(incar_output)

    @staticmethod
    def get_basis(structure: Structure, potcar_symbols: list,
                  address_basis_file: str = os.path.join(MODULE_DIR, "BASIS_PBE_54_standard.yaml")):
        """
        will get the basis from given potcar_symbols (e.g., ["Fe_pv","Si"]
        #include this in lobsterin class
        Args:
            structure (Structure): Structure object
            potcar_symbols: list of potcar symbols
        Returns:
            returns basis
        """
        Potcar_names = list(potcar_symbols)

        AtomTypes_Potcar = [name.split('_')[0] for name in Potcar_names]

        AtomTypes = structure.symbol_set

        if set(AtomTypes) != set(AtomTypes_Potcar):
            raise IOError("Your POSCAR does not correspond to your POTCAR!")
        BASIS = loadfn(address_basis_file)['BASIS']

        basis_functions = []
        list_forin = []
        for itype, type in enumerate(Potcar_names):
            if type not in BASIS:
                raise ValueError("You have to provide the basis for" + str(
                    type) + "manually. We don't have any information on this POTCAR.")
            basis_functions.append(BASIS[type].split())
            tojoin = str(AtomTypes_Potcar[itype]) + " "
            tojoin2 = "".join(str(str(e) + " ") for e in BASIS[type].split())
            list_forin.append(str(tojoin + tojoin2))
        return list_forin

    @staticmethod
    def get_all_possible_basis_functions(structure: Structure,
                                         potcar_symbols: list,
                                         address_basis_file_min:
                                         str = os.path.join(MODULE_DIR,
                                                            "lobster_basis/BASIS_PBE_54_min.yaml"),
                                         address_basis_file_max:
                                         str = os.path.join(MODULE_DIR,
                                                            "lobster_basis/BASIS_PBE_54_max.yaml")):

        """

        Args:
            structure: Structure object
            potcar_symbols: list of the potcar symbols
            address_basis_file_min: path to file with the minium required basis by the POTCAR
            address_basis_file_max: path to file with the largest possible basis of the POTCAR

        Returns: List of dictionaries that can be used to create new Lobsterin objects in
        standard_calculations_from_vasp_files as dict_for_basis

        """
        max_basis = Lobsterin.get_basis(structure=structure, potcar_symbols=potcar_symbols,
                                        address_basis_file=address_basis_file_max)
        min_basis = Lobsterin.get_basis(structure=structure, potcar_symbols=potcar_symbols,
                                        address_basis_file=address_basis_file_min)
        all_basis = get_all_possible_basis_combinations(min_basis=min_basis, max_basis=max_basis)
        list_basis_dict = []
        for ibasis, basis in enumerate(all_basis):
            basis_dict = {}

            for iel, elba in enumerate(basis):
                basplit = elba.split()
                basis_dict[basplit[0]] = " ".join(basplit[1:])
            list_basis_dict.append(basis_dict)
        return list_basis_dict

    @staticmethod
    def write_POSCAR_with_standard_primitive(POSCAR_input="POSCAR", POSCAR_output="POSCAR.lobster", symprec=0.01):
        """
        writes a POSCAR with the standard primitive cell. This is needed to arrive at the correct kpath
        Args:
            POSCAR_input (str): filename of input POSCAR
            POSCAR_output (str): filename of output POSCAR
            symprec (float): precision to find symmetry
        """
        structure = Structure.from_file(POSCAR_input)
        kpath = HighSymmKpath(structure, symprec=symprec)
        new_structure = kpath.prim
        new_structure.to(fmt='POSCAR', filename=POSCAR_output)

    @staticmethod
    def write_KPOINTS(POSCAR_input: str = "POSCAR", KPOINTS_output="KPOINTS.lobster", reciprocal_density: int = 100,
                      isym: int = -1, from_grid: bool = False, input_grid: list = [5, 5, 5], line_mode: bool = True,
                      kpoints_line_density: int = 20, symprec: float = 0.01):
        """
        writes a KPOINT file for lobster (only ISYM=-1 and ISYM=0 are possible), grids are gamma centered
        Args:
            POSCAR_input (str): path to POSCAR
            KPOINTS_output (str): path to output KPOINTS
            reciprocal_density (int): Grid density
            isym (int): either -1 or 0. Current Lobster versions only allow -1.
            from_grid (bool): If True KPOINTS will be generated with the help of a grid given in input_grid. Otherwise,
                they will be generated from the reciprocal_density
            input_grid (list): grid to generate the KPOINTS file
            line_mode (bool): If True, band structure will be generated
            kpoints_line_density (int): density of the lines in the band structure
            symprec (float): precision to determine symmetry
        """
        structure = Structure.from_file(POSCAR_input)
        if not from_grid:
            kpointgrid = Kpoints.automatic_density_by_vol(structure, reciprocal_density).kpts
            mesh = kpointgrid[0]
        else:
            mesh = input_grid

        # The following code is taken from: SpacegroupAnalyzer
        # we need to switch off symmetry here
        latt = structure.lattice.matrix
        positions = structure.frac_coords
        unique_species = []  # type: List[Any]
        zs = []
        magmoms = []

        for species, g in itertools.groupby(structure,
                                            key=lambda s: s.species):
            if species in unique_species:
                ind = unique_species.index(species)
                zs.extend([ind + 1] * len(tuple(g)))
            else:
                unique_species.append(species)
                zs.extend([len(unique_species)] * len(tuple(g)))

        for site in structure:
            if hasattr(site, 'magmom'):
                magmoms.append(site.magmom)
            elif site.is_ordered and hasattr(site.specie, 'spin'):
                magmoms.append(site.specie.spin)
            else:
                magmoms.append(0)

        # For now, we are setting magmom to zero. (Taken from INCAR class)
        cell = latt, positions, zs, magmoms
        # TODO: what about this shift?
        mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])

        # exit()
        # get the kpoints for the grid
        if isym == -1:
            kpts = []
            weights = []
            all_labels = []
            for gp in grid:
                kpts.append(gp.astype(float) / mesh)
                weights.append(float(1))
                all_labels.append("")
        elif isym == 0:
            # time reversal symmetry: k and -k are equivalent
            kpts = []
            weights = []
            all_labels = []
            newlist = [list(gp) for gp in list(grid)]
            mapping = []
            for gp in newlist:
                minusgp = [-k for k in gp]
                if minusgp in newlist and minusgp not in [[0, 0, 0]]:
                    mapping.append(newlist.index(minusgp))
                else:
                    mapping.append(newlist.index(gp))

            for igp, gp in enumerate(newlist):
                if mapping[igp] > igp:
                    kpts.append(np.array(gp).astype(float) / mesh)
                    weights.append(float(2))
                    all_labels.append("")
                elif mapping[igp] == igp:
                    kpts.append(np.array(gp).astype(float) / mesh)
                    weights.append(float(1))
                    all_labels.append("")

        else:
            ValueError("Only isym=-1 and isym=0 are allowed.")
        # line mode
        if line_mode:
            kpath = HighSymmKpath(structure, symprec=symprec)
            if not np.allclose(kpath.prim.lattice.matrix, structure.lattice.matrix):
                raise ValueError(
                    "You are not using the standard primitive cell. The k-path is not correct. Please generate a "
                    "standard primitive cell first.")

            frac_k_points, labels = kpath.get_kpoints(
                line_density=kpoints_line_density,
                coords_are_cartesian=False)

            for k, f in enumerate(frac_k_points):
                kpts.append(f)
                weights.append(0.0)
                all_labels.append(labels[k])
        if isym == -1:
            comment = (
                "ISYM=-1, grid: " + str(mesh) if not line_mode else "ISYM=-1, grid: " + str(mesh) + " plus kpoint path")
        elif isym == 0:
            comment = (
                "ISYM=0, grid: " + str(mesh) if not line_mode else "ISYM=0, grid: " + str(mesh) + " plus kpoint path")

        KpointObject = Kpoints(comment=comment,
                               style=Kpoints.supported_modes.Reciprocal,
                               num_kpts=len(kpts), kpts=kpts, kpts_weights=weights,
                               labels=all_labels)

        KpointObject.write_file(filename=KPOINTS_output)

    @classmethod
    def from_file(cls, lobsterin: str):
        """
        Args:
            lobsterin (str): path to lobsterin

        Returns:
            Lobsterin object
        """
        with zopen(lobsterin, 'rt') as f:
            data = f.read().split("\n")
        if len(data) == 0:
            raise IOError("lobsterin file contains no data.")
        Lobsterindict = {}  # type: Dict

        for datum in data:
            # will remove all commments to avoid complications
            raw_datum = datum.split('!')[0]
            raw_datum = raw_datum.split('//')[0]
            raw_datum = raw_datum.split('#')[0]
            raw_datum = raw_datum.split(' ')
            while "" in raw_datum:
                raw_datum.remove("")
            if len(raw_datum) > 1:
                # check which type of keyword this is, handle accordingly
                if raw_datum[0].lower() not in [datum2.lower() for datum2 in Lobsterin.LISTKEYWORDS]:
                    if raw_datum[0].lower() not in [datum2.lower() for datum2 in Lobsterin.FLOATKEYWORDS]:
                        if raw_datum[0].lower() not in Lobsterindict:
                            Lobsterindict[raw_datum[0].lower()] = " ".join(raw_datum[1:])
                        else:
                            raise ValueError("Same keyword " + str(raw_datum[0].lower()) + "twice!")
                    else:
                        if raw_datum[0].lower() not in Lobsterindict:
                            Lobsterindict[raw_datum[0].lower()] = float(raw_datum[1])
                        else:
                            raise ValueError("Same keyword " + str(raw_datum[0].lower()) + "twice!")
                else:
                    if raw_datum[0].lower() not in Lobsterindict:
                        Lobsterindict[raw_datum[0].lower()] = [" ".join(raw_datum[1:])]
                    else:
                        Lobsterindict[raw_datum[0].lower()].append(" ".join(raw_datum[1:]))
            elif len(raw_datum) > 0:
                Lobsterindict[raw_datum[0].lower()] = True

        return cls(Lobsterindict)

    @staticmethod
    def _get_potcar_symbols(POTCAR_input: str) -> list:
        """
        will return the name of the species in the POTCAR
        Args:
         POTCAR_input(str): string to potcar file
        Returns:
            list of the names of the species in string format
        """
        potcar = Potcar.from_file(POTCAR_input)
        # for pot in potcar:
        #     if pot.potential_type != "PAW":
        #         raise IOError("Lobster only works with PAW! Use different POTCARs")

        if potcar.functional != "PBE":
            raise IOError("We only have BASIS options for PBE so far")

        Potcar_names = [name.symbol for name in potcar]
        return Potcar_names

    @classmethod
    def standard_calculations_from_vasp_files(cls, POSCAR_input: str = "POSCAR", INCAR_input: str = "INCAR",
                                              POTCAR_input: Optional[str] = None,
                                              dict_for_basis: Optional[dict] = None,
                                              option: str = 'standard'):
        """
        will generate Lobsterin with standard settings

        Args:
            POSCAR_input(str): path to POSCAR
            INCAR_input(str): path to INCAR
            POTCAR_input (str): path to POTCAR
            dict_for_basis (dict): can be provided: it should look the following:
                dict_for_basis={"Fe":'3p 3d 4s 4f', "C": '2s 2p'} and will overwrite all settings from POTCAR_input

            option (str): 'standard' will start a normal lobster run where COHPs, COOPs, DOS, CHARGE etc. will be
                calculated
                'standard_from_projection' will start a normal lobster run from a projection
                'standard_with_fatband' will do a fatband calculation, run over all orbitals
                'onlyprojection' will only do a projection
                'onlydos' will only calculate a projected dos
                'onlycohp' will only calculate cohp
                'onlycoop' will only calculate coop
                'onlycohpcoop' will only calculate cohp and coop

        Returns:
            Lobsterin Object with standard settings
        """
        # warn that fatband calc cannot be done with tetrahedron method at the moment
        if option not in ['standard', 'standard_from_projection', 'standard_with_fatband', 'onlyprojection', 'onlydos',
                          'onlycohp', 'onlycoop', 'onlycohpcoop']:
            raise ValueError("The option is not valid!")

        Lobsterindict = {}  # type: Dict[Any,Any]
        # this basis set covers most elements
        Lobsterindict['basisSet'] = 'pbeVaspFit2015'
        # energies around e-fermi
        Lobsterindict['COHPstartEnergy'] = -15.0
        Lobsterindict['COHPendEnergy'] = 5.0

        if option in ['standard', 'onlycohp', 'onlycoop', 'onlycohpcoop', 'standard_with_fatband']:
            # every interaction with a distance of 6.0 is checked
            Lobsterindict['cohpGenerator'] = "from 0.1 to 6.0 orbitalwise"
            # the projection is saved
            Lobsterindict['saveProjectionToFile'] = True

        if option == 'standard_from_projection':
            Lobsterindict['cohpGenerator'] = "from 0.1 to 6.0 orbitalwise"
            Lobsterindict['loadProjectionFromFile'] = True

        if option == 'onlycohp':
            Lobsterindict['skipdos'] = True
            Lobsterindict['skipcoop'] = True
            Lobsterindict['skipPopulationAnalysis'] = True
            Lobsterindict['skipGrossPopulation'] = True

        if option == 'onlycoop':
            Lobsterindict['skipdos'] = True
            Lobsterindict['skipcohp'] = True
            Lobsterindict['skipPopulationAnalysis'] = True
            Lobsterindict['skipGrossPopulation'] = True

        if option == 'onlycohpcoop':
            Lobsterindict['skipdos'] = True
            Lobsterindict['skipPopulationAnalysis'] = True
            Lobsterindict['skipGrossPopulation'] = True

        if option == 'onlydos':
            Lobsterindict['skipcohp'] = True
            Lobsterindict['skipcoop'] = True
            Lobsterindict['skipPopulationAnalysis'] = True
            Lobsterindict['skipGrossPopulation'] = True

        if option == 'onlyprojection':
            Lobsterindict['skipdos'] = True
            Lobsterindict['skipcohp'] = True
            Lobsterindict['skipcoop'] = True
            Lobsterindict['skipPopulationAnalysis'] = True
            Lobsterindict['skipGrossPopulation'] = True
            Lobsterindict['saveProjectionToFile'] = True

        incar = Incar.from_file(INCAR_input)
        if incar["ISMEAR"] == 0:
            Lobsterindict['gaussianSmearingWidth'] = incar["SIGMA"]
        if incar["ISMEAR"] != 0 and option == "standard_with_fatband":
            raise ValueError("ISMEAR has to be 0 for a fatband calculation with Lobster")
        if dict_for_basis is not None:
            # dict_for_basis={"Fe":'3p 3d 4s 4f', "C": '2s 2p'}
            # will just insert this basis and not check with poscar
            basis = [key + ' ' + value for key, value in dict_for_basis.items()]
        elif POTCAR_input is not None:
            # get basis from POTCAR
            potcar_names = Lobsterin._get_potcar_symbols(POTCAR_input=POTCAR_input)

            basis = Lobsterin.get_basis(structure=Structure.from_file(POSCAR_input),
                                        potcar_symbols=potcar_names)
        else:
            raise ValueError("basis cannot be generated")
        Lobsterindict["basisfunctions"] = basis
        if option == 'standard_with_fatband':
            Lobsterindict['createFatband'] = basis

        return cls(Lobsterindict)


def get_all_possible_basis_combinations(min_basis: list, max_basis: list) -> list:
    """

    Args:
        min_basis: list of basis entries: e.g., ['Si 3p 3s ']
        max_basis: list of basis entries: e.g., ['Si 3p 3s ']

    Returns: all possible combinations of basis functions, e.g. [['Si 3p 3s']]

    """
    max_basis_lists = [x.split() for x in max_basis]
    min_basis_lists = [x.split() for x in min_basis]

    # get all possible basis functions
    basis_dict = collections.OrderedDict({})  # type:  Dict[Any, Any]
    for iel, el in enumerate(max_basis_lists):
        basis_dict[el[0]] = {"fixed": [], "variable": [], "combinations": []}
        for basis in el[1:]:
            if basis in min_basis_lists[iel]:
                basis_dict[el[0]]["fixed"].append(basis)
            if basis not in min_basis_lists[iel]:
                basis_dict[el[0]]["variable"].append(basis)
        for L in range(0, len(basis_dict[el[0]]['variable']) + 1):
            for subset in itertools.combinations(basis_dict[el[0]]['variable'], L):
                basis_dict[el[0]]["combinations"].append(' '.join([el[0]] + basis_dict[el[0]]['fixed'] + list(subset)))

    list_basis = []
    for el, item in basis_dict.items():
        list_basis.append(item['combinations'])

    # get all combinations
    start_basis = list_basis[0]
    if len(list_basis) > 1:
        for iel, el in enumerate(list_basis[1:], 1):
            new_start_basis = []
            for ielbasis, elbasis in enumerate(start_basis):
                for ielbasis2, elbasis2 in enumerate(list_basis[iel]):
                    if not isinstance(elbasis, list):
                        new_start_basis.append([elbasis, elbasis2])
                    else:
                        new_start_basis.append(elbasis.copy() + [elbasis2])
            start_basis = new_start_basis
        return start_basis
    return [[basis] for basis in start_basis]
