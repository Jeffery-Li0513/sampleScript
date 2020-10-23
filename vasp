    def from_file(cls, filename, primitive=False, sort=False, merge_tol=0.0):
        """
        Reads a structure from a file. For example, anything ending in
        a "cif" is assumed to be a Crystallographic Information Format file.
        Supported formats include CIF, POSCAR/CONTCAR, CHGCAR, LOCPOT,
        vasprun.xml, CSSR, Netcdf and pymatgen's JSON serialized structures.

        Args:
            filename (str): The filename to read from.
            primitive (bool): Whether to convert to a primitive cell
                Only available for cifs. Defaults to False.
            sort (bool): Whether to sort sites. Default to False.
            merge_tol (float): If this is some positive number, sites that
                are within merge_tol from each other will be merged. Usually
                0.01 should be enough to deal with common numerical issues.

        Returns:
            Structure.
        """
        filename = str(filename)
        if filename.endswith(".nc"):
            # Read Structure from a netcdf file.
            from pymatgen.io.abinit.netcdf import structure_from_ncdata
            s = structure_from_ncdata(filename, cls=cls)
            if sort:
                s = s.get_sorted_structure()
            return s

        from pymatgen.io.lmto import LMTOCtrl
        from pymatgen.io.vasp import Vasprun, Chgcar
        from pymatgen.io.exciting import ExcitingInput
        from monty.io import zopen
        fname = os.path.basename(filename)
        with zopen(filename, "rt") as f:
            contents = f.read()
        if fnmatch(fname.lower(), "*.cif*") or fnmatch(fname.lower(), "*.mcif*"):
            return cls.from_str(contents, fmt="cif",
                                primitive=primitive, sort=sort,
                                merge_tol=merge_tol)
        elif fnmatch(fname, "*POSCAR*") or fnmatch(fname, "*CONTCAR*") or fnmatch(fname, "*.vasp"):
            s = cls.from_str(contents, fmt="poscar",
                             primitive=primitive, sort=sort,
                             merge_tol=merge_tol)

        elif fnmatch(fname, "CHGCAR*") or fnmatch(fname, "LOCPOT*"):
            s = Chgcar.from_file(filename).structure
        elif fnmatch(fname, "vasprun*.xml*"):
            s = Vasprun(filename).final_structure
        elif fnmatch(fname.lower(), "*.cssr*"):
            return cls.from_str(contents, fmt="cssr",
                                primitive=primitive, sort=sort,
                                merge_tol=merge_tol)
        elif fnmatch(fname, "*.json*") or fnmatch(fname, "*.mson*"):
            return cls.from_str(contents, fmt="json",
                                primitive=primitive, sort=sort,
                                merge_tol=merge_tol)
        elif fnmatch(fname, "*.yaml*"):
            return cls.from_str(contents, fmt="yaml",
                                primitive=primitive, sort=sort,
                                merge_tol=merge_tol)
        elif fnmatch(fname, "*.xsf"):
            return cls.from_str(contents, fmt="xsf",
                                primitive=primitive, sort=sort,
                                merge_tol=merge_tol)
        elif fnmatch(fname, "input*.xml"):
            return ExcitingInput.from_file(fname).structure
        elif fnmatch(fname, "*rndstr.in*") \
                or fnmatch(fname, "*lat.in*") \
                or fnmatch(fname, "*bestsqs*"):
            return cls.from_str(contents, fmt="mcsqs",
                                primitive=primitive, sort=sort,
                                merge_tol=merge_tol)
        elif fnmatch(fname, "CTRL*"):
            return LMTOCtrl.from_file(filename=filename).structure
        else:
            raise ValueError("Unrecognized file extension!")
        if sort:
            s = s.get_sorted_structure()
        if merge_tol:
            s.merge_sites(merge_tol)

        s.__class__ = cls
        return s
