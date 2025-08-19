import pandas as pd
import os
import tarfile
import gzip
import shutil
import json
import glob
import time
import re
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, MMCIFParser, Select, PDBIO

index_file = '/mnt/research/woldring_lab/Members/Eaves/plip-plop/index/PDBbind_v2020_core_info.csv'
index_df = pd.read_csv(index_file)
base_dir = '/mnt/research/woldring_lab/Members/Eaves/plip-plop/data'
model_dir = '/mnt/research/woldring_lab/Members/Eaves/plip-plop/OnionNet-2/models'
scaler_fpath = os.path.join(model_dir, 'train_scaler.scaler')
model_fpath = os.path.join(model_dir, '62shell_saved-model.h5')
out_csv = '/mnt/research/woldring_lab/Members/Eaves/plip-plop/results/CASF2016_OnionNet2.csv'
timings_log = '/mnt/research/woldring_lab/Members/Eaves/plip-plop/results/onionnet2_timings.jsonl'
sources = ['gnina', 'rosetta', 'boltz2']
n_poses = 10

ONIONNET2_INFER = '/mnt/research/woldring_lab/Members/Eaves/plip-plop/OnionNet-2/scoring/predict.py'

PROT_RES3 = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
    'ASX','GLX','SEC','PYL'
}
WAT = {'HOH','WAT'}

TIMES = {}

def tick():
    return time.perf_counter()

def tock(key, t0):
    TIMES[key] = TIMES.get(key, 0.0) + (time.perf_counter() - t0)
    return TIMES[key]

def log_timing(pdb_id, source, name, step, seconds, extra=None):
    rec = {'pdb_id': pdb_id, 'source': source, 'name': name, 'step': step, 'seconds': round(float(seconds), 6)}
    if extra:
        rec.update(extra)
    os.makedirs(os.path.dirname(timings_log), exist_ok=True)
    with open(timings_log, 'a') as f:
        f.write(json.dumps(rec) + '\n')

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def gunzip_to_tmp(gz_path):
    tmp_path = gz_path[:-3]
    with gzip.open(gz_path, 'rb') as f_in:
        with open(tmp_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return tmp_path

class ProteinOnlySelect(Select):
    def accept_residue(self, residue):
        return residue.id[0] == ' '
    
def sdf_to_pdb(sdf_path, pdb_path):
    subprocess.run(['obabel', sdf_path, '-O', pdb_path])

def write_ligand_files(lig_mol, out_dir, stem):
    sdf_path = os.path.join(out_dir, f'{stem}.sdf')
    pdb_path = os.path.join(out_dir, f'{stem}.pdb')
    w = Chem.SDWriter(sdf_path)
    w.write(lig_mol)
    w.close()
    Chem.MolToPDBFile(lig_mol, pdb_path)
    return sdf_path, pdb_path

def remap_paths_if_needed(source, archive_path, complex_path, score_file):
    new_complex = complex_path
    new_score = score_file
    # If the expected directory structure is different after extraction, find the real paths
    if source == 'gnina':
        # Try to locate any file that matches the pose template with POSE replaced by wildcard
        base_name = os.path.basename(complex_path)
        if 'POSE' in base_name:
            pat = os.path.join(archive_path, '**', base_name.replace('POSE', '*'))
            hits = glob.glob(pat, recursive=True)
            if hits:
                pose_dir = os.path.dirname(hits[0])
                new_complex = os.path.join(pose_dir, base_name)
    # Remap score file if it is missing
    if score_file and not os.path.exists(score_file):
        score_base = os.path.basename(score_file)
        pat = os.path.join(archive_path, '**', score_base)
        hits = glob.glob(pat, recursive=True)
        if hits:
            new_score = hits[0]
    return new_complex, new_score

def _pdb_element_from_name(name):
    # derive element symbol from atom name per PDB convention
    s = re.sub('[^A-Za-z]', '', str(name or '')).upper()
    if not s:
        return 'C'
    if s.startswith('CL'):  # common 2-letter halogens first
        return 'CL'
    if s.startswith('BR'):
        return 'BR'
    return s[0]

def _format_pdb_atom_line(record, serial, name, resName, chainID, resSeq,
                          x, y, z, occupancy=1.00, tempFactor=0.00,
                          element=None, altLoc=' ', iCode=' ', charge='  '):
    # record: 'ATOM  ' or 'HETATM'
    # PDB fixed-width columns per spec; RDKit relies on these positions
    # 1-6  record, 7-11 serial, 13-16 name, 17 altLoc, 18-20 resName, 22 chainID,
    # 23-26 resSeq, 27 iCode, 31-38 x, 39-46 y, 47-54 z, 55-60 occ, 61-66 temp,
    # 77-78 element, 79-80 charge
    if element is None:
        element = _pdb_element_from_name(name)
    element = element.upper()
    nm = str(name or '').strip()
    # PDB name placement rule:
    # - 2-letter element: name is left-justified in cols 13-16
    # - 1-letter element: name is right-justified in cols 14-16 with a leading space
    if len(element) == 2:
        name_fmt = f'{nm:<4.4s}'
    else:
        name_fmt = f' {nm:>3.3s}'
    line = (
        f'{record:<6s}'
        f'{serial:>5d} '
        f'{name_fmt}'
        f'{altLoc:1s}'
        f'{resName:>3s} '
        f'{chainID:1s}'
        f'{resSeq:>4d}'
        f'{iCode:1s}'
        f'   '
        f'{x:>8.3f}'
        f'{y:>8.3f}'
        f'{z:>8.3f}'
        f'{occupancy:>6.2f}'
        f'{tempFactor:>6.2f}'
        f'          '
        f'{element:>2s}'
        f'{charge:>2s}'
    )
    return line

def extract_protein_and_ligand_from_complex_pdb(pdb_path, out_dir):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_path)
    io = PDBIO()
    prot_path = os.path.join(out_dir, 'protein_only.pdb')
    io.set_structure(structure)
    io.save(prot_path, ProteinOnlySelect())

    lig_records = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in WAT:
                    continue
                if resname not in PROT_RES3:
                    lig_records.append((len(list(residue.get_atoms())), model.id, chain.id, residue.id, resname))
    if len(lig_records) == 0:
        for model in structure:
            for chain in model:
                for residue in chain:
                    hetflag = residue.id[0].strip()
                    resname = residue.get_resname().strip()
                    if hetflag != '' and resname not in WAT:
                        lig_records.append((len(list(residue.get_atoms())), model.id, chain.id, residue.id, resname))
    if len(lig_records) == 0:
        raise ValueError('no ligand-like residue found in complex pdb')

    lig_records.sort(key=lambda x: x[0], reverse=True)
    _, model_id, chain_id, resid, resname = lig_records[0]

    lig_atoms = []
    for model in structure:
        if model.id != model_id:
            continue
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id == resid:
                    for atom in residue.get_atoms():
                        lig_atoms.append(atom)
                    break

    lig_pdb_path = os.path.join(out_dir, 'ligand_only.pdb')
    with open(lig_pdb_path, 'w') as f:
        serial = 1
        ch = chain_id if isinstance(chain_id, str) and len(chain_id) == 1 else 'A'
        resseq = 1
        for atom in lig_atoms:
            name = atom.get_name().rjust(4)
            rname = resname.rjust(3)
            x, y, z = atom.coord
            occ = atom.get_occupancy() if atom.get_occupancy() is not None else 1.0
            b = atom.get_bfactor() if atom.get_bfactor() is not None else 0.0
            elem = (atom.element.strip() if atom.element else 'C')[:2].rjust(2)
            f.write(f'HETATM{serial:5d} {name} {rname} {ch}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {elem}\n')
            serial += 1
        f.write('END\n')

    lig_mol = Chem.MolFromPDBFile(lig_pdb_path, removeHs=False)
    if lig_mol is None:
        lig_mol = Chem.MolFromPDBFile(lig_pdb_path, removeHs=True)
    if lig_mol is None:
        with open(lig_pdb_path, 'r') as fh:
            pdb_block = fh.read()
        lig_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, proximityBonding=True, sanitize=False)
        if lig_mol is not None:
            try:
                Chem.SanitizeMol(lig_mol)
            except Exception:
                pass
    if lig_mol is None:
        raise ValueError('failed to parse ligand from complex pdb')
    lig_mol = Chem.AddHs(lig_mol, addCoords=True)
    if lig_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(lig_mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(lig_mol, maxIters=200)
    return prot_path, lig_mol

def convert_modelcif_to_pdb(cif_path, pdb_path):
    subprocess.run(['obabel', '-icif', cif_path, '-O', pdb_path])

def extract_protein_and_ligand_from_complex_cif(cif_path, out_dir):
    tmp_pdb = os.path.join(out_dir, 'converted_from_cif.pdb')
    convert_modelcif_to_pdb(cif_path, tmp_pdb)
    return extract_protein_and_ligand_from_complex_pdb(tmp_pdb, out_dir)

def onionnet2_predict(rec_pdb, lig_pdb, scaler, model, out_csv, name_tag):
    if not os.path.exists(ONIONNET2_INFER):
        raise FileNotFoundError('set ONIONNET2_INFER to OnionNet-2 inference script')
    cmd = [
        'python3', ONIONNET2_INFER,
        '-rec_fpath', rec_pdb,
        '-lig_fpath', lig_pdb,
        '-shape', '84,124,1',
        '-scaler', scaler,
        '-model', model,
        '-shells', '62',
        '-out_fpath', out_csv,
        '-name', name_tag
    ]
    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', '')
    env.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # reduce TF logs
    subprocess.run(cmd, check=True, env=env)

def archive_with_pose(archive_dir, pose_src_path, protein_pdb, ligand_sdf, ligand_pdb, name_tag):
    ensure_dir(archive_dir)
    tar_path = os.path.join(archive_dir, f'{name_tag}.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tf:
        tf.add(pose_src_path, arcname=os.path.basename(pose_src_path))
        tf.add(protein_pdb, arcname='protein_from_complex.pdb')
        tf.add(ligand_sdf, arcname='ligand_from_complex.sdf')
        tf.add(ligand_pdb, arcname='ligand_from_complex.pdb')
    return tar_path

def process_pose(complex_path, source, name_tag, work_dir, archive_dir, pdb_id, source_tar_path=None):
    tmp_cleanup = None
    use_path = complex_path
    if use_path.endswith('.gz'):
        use_path = gunzip_to_tmp(use_path)
        tmp_cleanup = use_path

    ensure_dir(work_dir)
    t0 = tick()
    try:
        if source in ('gnina', 'rosetta'):
            prot_pdb, lig_mol = extract_protein_and_ligand_from_complex_pdb(use_path, work_dir)
        elif source == 'boltz2':
            prot_pdb, lig_mol = extract_protein_and_ligand_from_complex_cif(use_path, work_dir)
        else:
            raise ValueError('unknown source for pose processing')
        dt = time.perf_counter() - t0
        log_timing(pdb_id, source, name_tag, 'split_complex', dt)
    except Exception as e:
        if tmp_cleanup and os.path.exists(tmp_cleanup):
            os.remove(tmp_cleanup)
        raise

    sdf_path, lig_pdb_path = write_ligand_files(lig_mol, work_dir, 'ligand_from_complex')

    try:
        t1 = tick()
        onionnet2_predict(prot_pdb, lig_pdb_path, scaler_fpath, model_fpath, out_csv, name_tag)
        dt_inf = time.perf_counter() - t1
        log_timing(pdb_id, source, name_tag, 'onionnet2_infer', dt_inf)
    finally:
        if tmp_cleanup and os.path.exists(tmp_cleanup):
            os.remove(tmp_cleanup)

    # per-pose archive of pose + intermediates for your own records
    t_arch = tick()
    tar_path = archive_with_pose(archive_dir, complex_path, prot_pdb, sdf_path, lig_pdb_path, name_tag)
    dt_arch = time.perf_counter() - t_arch
    log_timing(pdb_id, source, name_tag, 'archive_with_pose', dt_arch)

    # also append intermediates into the original docking tar, then clean up
    try:
        if source_tar_path:
            archive_intermediates_into_source_tar(source_tar_path, name_tag, prot_pdb, sdf_path, lig_pdb_path)
        for f in (prot_pdb, sdf_path, lig_pdb_path):
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f'[WARNING] failed to remove intermediate {f}: {e}')
    except Exception as e:
        print(f'[WARNING] failed to append intermediates to source tar: {e}')

    # clean up per-pose working directory after we are done
    try:
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
    except Exception as e:
        print(f'[WARNING] failed to remove work dir {work_dir}: {e}')

    return {'name': name_tag, 'tar': tar_path, 'rec': prot_pdb, 'lig_sdf': sdf_path, 'lig_pdb': lig_pdb_path}

def _repack_tar_gz_with_additions(src_tar_gz, files_to_add, arc_prefix):
    # tarfile cannot reliably append to gz archives; rebuild safely
    dname = os.path.dirname(src_tar_gz)
    base = os.path.basename(src_tar_gz)
    tmpdir = tempfile.mkdtemp(prefix='repack_', dir=dname if dname else None)
    try:
        # extract original into tmpdir/orig
        orig_dir = os.path.join(tmpdir, 'orig')
        os.makedirs(orig_dir, exist_ok=True)
        with tarfile.open(src_tar_gz, 'r:gz') as tin:
            tin.extractall(path=orig_dir)
        # write new tar.gz including original content
        new_tar = os.path.join(tmpdir, base + '.new')
        with tarfile.open(new_tar, 'w:gz') as tout:
            for root, _, files in os.walk(orig_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    arcname = os.path.relpath(fp, orig_dir)
                    tout.add(fp, arcname=arcname)
            # add new files under arc_prefix
            for f in files_to_add:
                if f and os.path.exists(f):
                    arcname = os.path.join(arc_prefix, os.path.basename(f))
                    tout.add(f, arcname=arcname)
        # atomically replace original
        shutil.move(new_tar, src_tar_gz)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def archive_intermediates_into_source_tar(tar_path, name_tag, protein_pdb, ligand_sdf, ligand_pdb):
    if not tar_path or not os.path.exists(tar_path):
        print(f'[WARNING] source tar not found, skipping intermediate archival: {tar_path}')
        return
    arc_prefix = os.path.join('intermediates', name_tag)
    _repack_tar_gz_with_additions(
        src_tar_gz=tar_path,
        files_to_add=[protein_pdb, ligand_sdf, ligand_pdb],
        arc_prefix=arc_prefix
    )

def main():
    df = pd.read_csv(index_file)
    for pdb_id in df['pdb_id'].unique():

        pdb_df = df[df['pdb_id'] == pdb_id]
        
        # crystal
        protein_pdb = pdb_df['prot_path'].values[0]
        ligand_sdf = pdb_df['lig_path'].values[0]
        if os.path.exists(protein_pdb) and os.path.exists(ligand_sdf):
            lig_pdb = os.path.splitext(ligand_sdf)[0] + '.pdb'
            if not os.path.exists(lig_pdb):
                Chem.MolToPDBFile(Chem.SDMolSupplier(ligand_sdf, removeHs=False)[0], lig_pdb)
            name = f'{pdb_id}-crystal'
            try:
                t0 = tick()
                onionnet2_predict(protein_pdb, lig_pdb, scaler_fpath, model_fpath, out_csv, name)
                dt = time.perf_counter() - t0
                log_timing(pdb_id, 'crystal', name, 'onionnet2_infer', dt)
            except Exception as e:
                print(f'[WARNING] onionnet2 failed for crystal {pdb_id}: {e}')
        else:
            print(f'[WARNING] missing crystal files for {pdb_id}')

        # docked sources
        for source in sources:
            
            src_df = pdb_df[pdb_df['source'] == source]
            if src_df.empty:
                print(f'[WARNING] No entries for {pdb_id} {source}. Skipping.')
                continue

            tar_path = src_df['tar_path'].values[0]
            complex_path = src_df['archive_path'].values[0]
            score_file = src_df['score_file'].values[0]
            archive_path = os.path.dirname(complex_path)

            extracted_here = False
            if os.path.exists(tar_path):
                ensure_dir(archive_path)
                with tarfile.open(tar_path, 'r:*') as tar:
                    tar.extractall(path=archive_path)
                print(f'Extracted {source} archive to {archive_path}.')
                extracted_here = True
                complex_path, score_file = remap_paths_if_needed(source, archive_path, complex_path, score_file)
                print(f'Post-extract complex_path: {complex_path}')
                if score_file:
                    print(f'Post-extract score_file: {score_file}')

            if not os.path.isdir(archive_path):
                print(f'[WARNING] could not locate archive dir {archive_path} for {pdb_id} {source}')
                continue

            outdir = os.path.join(base_dir, pdb_id, 'onionnet2', source)
            archive_dir = os.path.join(base_dir, pdb_id, f'{source}_prepped_archives')
            ensure_dir(outdir)
            ensure_dir(archive_dir)

            try:
                if source == 'gnina':
                    for n in range(1, n_poses + 1):
                        pose_id = str(n).zfill(4)
                        p3 = complex_path.replace('POSE', str(n).zfill(3))
                        p4 = complex_path.replace('POSE', pose_id)
                        pose = p4 if os.path.exists(p4) else p3
                        if not os.path.exists(pose):
                            print(f'[WARNING] gnina pose {n} missing for {pdb_id}')
                            continue
                        name = f'{pdb_id}-gnina-{pose_id}'
                        work_dir = os.path.join(outdir, pose_id)
                        try:
                            process_pose(pose, 'gnina', name, work_dir, archive_dir, pdb_id, source_tar_path=tar_path)
                        except Exception as e:
                            print(f'[WARNING] failed gnina {name}: {e}')

                elif source == 'rosetta':
                    if not score_file or not os.path.exists(score_file):
                        print(f'[WARNING] missing rosetta score file for {pdb_id}')
                        continue
                    info = []
                    with open(score_file, 'r') as fh:
                        for line in fh:
                            row = json.loads(line)
                            info.append({'pose_id': row['decoy'][-4:], 'interface_delta_X': row['interface_delta_X']})
                    score_df = pd.DataFrame(info).sort_values(by='interface_delta_X').head(n_poses)
                    template = complex_path
                    first = str(int(score_df.iloc[0]['pose_id'])).zfill(4) if len(score_df) else None
                    if first:
                        probe = template.replace('POSE', first)
                        if not os.path.exists(probe):
                            pat = os.path.basename(template).replace('POSE', first)
                            hit = next(iter(glob.glob(os.path.join(archive_path, '**', pat), recursive=True)), None)
                            if hit:
                                template = os.path.join(os.path.dirname(hit), os.path.basename(template))
                                print(f'Resolved rosetta template to {template}')
                    for rank, pose_id in enumerate(score_df['pose_id'].tolist(), start=1):
                        pose = template.replace('POSE', str(int(pose_id)).zfill(4))
                        if not os.path.exists(pose):
                            print(f'[WARNING] rosetta pose {pose_id} missing for {pdb_id}')
                            continue
                        name = f'{pdb_id}-rosetta-{str(int(pose_id)).zfill(4)}'
                        work_dir = os.path.join(outdir, str(rank).zfill(2))
                        try:
                            process_pose(pose, 'rosetta', name, work_dir, archive_dir, pdb_id, source_tar_path=tar_path)
                        except Exception as e:
                            print(f'[WARNING] failed rosetta {name}: {e}')

                elif source == 'boltz2':
                    pose = complex_path
                    if not os.path.exists(pose):
                        base = os.path.basename(pose)
                        hit = next(iter(glob.glob(os.path.join(archive_path, '**', base), recursive=True)), None)
                        if not hit:
                            print(f'[WARNING] boltz2 cif missing for {pdb_id}')
                            continue
                        pose = hit
                    name = f'{pdb_id}-boltz2-model0'
                    work_dir = os.path.join(outdir, 'model0')
                    try:
                        process_pose(pose, 'boltz2', name, work_dir, archive_dir, pdb_id, source_tar_path=tar_path)
                    except Exception as e:
                        print(f'[WARNING] failed boltz2 {name}: {e}')

            finally:
                if extracted_here and os.path.exists(archive_path):
                    try:
                        shutil.rmtree(archive_path)
                    except Exception as e:
                        print(f'[WARNING] failed to remove extracted dir {archive_path}: {e}')

    # optional: print timings
    if TIMES:
        print('timings_seconds:', {k: round(v, 3) for k, v in TIMES.items()})

if __name__ == '__main__':
    main()