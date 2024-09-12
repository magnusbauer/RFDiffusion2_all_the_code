import torch
import sys
import numpy as np 
import random

from rf_diffusion.chemical import ChemicalData as ChemData


class ContigMap():
    '''
    New class for doing mapping.
    Supports multichain or multiple crops from a single chain.
    Also supports indexing jump (+200) or not, based on contig input.
    Default chain outputs are inpainted chains as A (and B, C etc if multiple chains)
    Output chains can be specified. Sequence must be the same number of elements as in contig string
    '''
    def __init__(
            self,
            parsed_pdb,
            contigs=None,
            contig_atoms=None,
            inpaint_seq=None,
            inpaint_str=None,
            length=None,
            has_termini=None,
            ref_idx=None,
            hal_idx=None,
            idx_rf=None,
            inpaint_seq_tensor=None,
            inpaint_str_tensor=None,
            topo=False,
            shuffle=False,
            intersperse='10-100',
            ):

        if shuffle:
            shuffled_contig_list = np.array(contigs[0].strip().split(','))
            print(f'{length=}')
            print(f'before shuffle: {contigs=}')
            print(f'{shuffled_contig_list=}')
            is_motif = np.array([e[0].isalpha() for e in shuffled_contig_list])
            motifs = shuffled_contig_list[is_motif]
            np.random.shuffle(motifs)
            shuffled_contig_list[is_motif] = motifs
            shuffled_contig_list[~is_motif] = intersperse
            contigs = [','.join(shuffled_contig_list)]
            print(f'after shuffle: {contigs=}')

        #sanity checks
        if contigs is None and ref_idx is None:
            sys.exit("Must either specify a contig string or precise mapping")
        if all([idx_rf is not None or hal_idx is not None or ref_idx is not None,
                idx_rf is None or hal_idx is None or ref_idx is None]):
            sys.exit("If you're specifying specific contig mappings, the reference and output positions must be specified, AND the indexing for RoseTTAFold (idx_rf)")

        self.chain_order='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if length is None:
            self.length = None
        elif '-' not in length:
            self.length = [int(length),int(length)+1]
        else:
            self.length = [int(length.split("-")[0]),int(length.split("-")[1])+1]
        self.has_termini=has_termini
        self.ref_idx = ref_idx
        self.hal_idx=hal_idx
        self.idx_rf=idx_rf
        self.inpaint_seq = ','.join(inpaint_seq).split(",") if inpaint_seq is not None else None
        self.inpaint_str = ','.join(inpaint_str).split(",") if inpaint_str is not None else None
        self.inpaint_seq_tensor=inpaint_seq_tensor
        self.inpaint_str_tensor=inpaint_str_tensor
        self.parsed_pdb = parsed_pdb
        self.topo=topo
        if ref_idx is None:
            #using default contig generation, which outputs in rosetta-like format
            self.contigs=contigs
            if contig_atoms is None:
                self.contig_atoms = None
            elif contig_atoms.find('-') > -1:
                # Use alternative reader
                self.contig_atoms={kv.split('-')[0]:kv.split("-")[1:] for kv in contig_atoms.split('_')}
            else:
                # Use standard
                self.contig_atoms={k:v.split(",") for k,v in eval(contig_atoms).items()}
                self.contig_atoms={k:[e for e in v if e != ''] for k,v in self.contig_atoms.items()}
            self.sampled_mask,self.contig_length,self.n_inpaint_chains = self.get_sampled_mask()
            self.inpaint, self.inpaint_hal, self.atomize_resnum2atomnames = self.expand_sampled_mask()
            self.ref = self.inpaint.copy()
            self.hal = self.inpaint_hal.copy()
            self.atomize_indices2atomname = {self.ref.index(res_num):atom_names for res_num, atom_names in self.atomize_resnum2atomnames.items()}
            self.atomize_indices = list(self.atomize_indices2atomname.keys())
        else:
            #specifying precise mappings
            self.ref=ref_idx
            self.hal=hal_idx

        self.mask_1d = [bool(i != ('_','_')) for i in self.ref]
        #take care of sequence and structure masking
        if self.inpaint_seq_tensor is None:
            if self.inpaint_seq is not None:
                self.inpaint_seq = self.get_inpaint_seq_str(self.inpaint_seq)
            else:
                self.inpaint_seq = np.array([bool(i != ('_','_')) for i in self.ref])
        else:
            self.inpaint_seq = self.inpaint_seq_tensor

        if self.inpaint_str_tensor is None:
            if self.inpaint_str is not None:
                self.inpaint_str = self.get_inpaint_seq_str(self.inpaint_str)
            else:
                self.inpaint_str = np.array([bool(i != ('_','_')) for i in self.ref])
        else:
            self.inpaint_str = self.inpaint_str_tensor
        #get 0-indexed input/output (for trb file)
        self.ref_idx0,self.hal_idx0, self.ref_idx0_inpaint, self.hal_idx0_inpaint=self.get_idx0()
        self.con_ref_pdb_idx=[i for i in self.ref if i != ('_','_')]


    def get_sampled_mask(self):
        '''
        Function to get a sampled mask from a contig.
        '''
        length_compatible=False
        count = 0
        while not length_compatible:
            inpaint_chains=0
            contig_list = self.contigs[0].strip().split('_')
            sampled_mask = []
            sampled_mask_length = 0
            for con in contig_list:
                inpaint_chains += 1
                #chain to be inpainted. These are the only chains that count towards the length of the contig
                subcons = con.split(",")
                subcon_out = []
                for subcon in subcons:
                    if subcon[0].isalpha():
                        subcon_out.append(subcon)
                        if '-' in subcon:
                            sampled_mask_length += (int(subcon.split("-")[1])-int(subcon.split("-")[0][1:])+1)
                        else:
                            sampled_mask_length += 1

                    else:
                        if '-' in subcon:
                            length_inpaint=random.randint(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += length_inpaint
                        elif subcon == '0':
                            subcon_out.append('0')
                        else:
                            length_inpaint=int(subcon)
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += int(subcon)
                sampled_mask.append(','.join(subcon_out))
            #check length is compatible
            if self.length is None:
                length_compatible = True
            elif sampled_mask_length >= self.length[0] and sampled_mask_length < self.length[1]:
                length_compatible = True
            count+=1
            if count == 100000: #contig string incompatible with this length
                sys.exit("Contig string incompatible with --length range")
        return sampled_mask, sampled_mask_length, inpaint_chains
    
    def seq(self, chain, residue_idx):
        seq_by_chain_resi = dict(zip(self.parsed_pdb['pdb_idx'], self.parsed_pdb['seq']))
        return seq_by_chain_resi[(chain, residue_idx)]
    
    def atom_names(self, chain, residue_idx):
        seq_token = self.seq(chain, residue_idx)
        atom_names = [atom_name.strip() for atom_name in ChemData().aa2long[seq_token][:ChemData().NHEAVY] if atom_name is not None]
        return atom_names

    def expand_sampled_mask(self):
        chain_order='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        chains=[]
        inpaint = []
        inpaint_hal = []
        inpaint_idx = 1
        inpaint_chain_idx=-1

        atomize_resnum2atomnames = {}
        for con in self.sampled_mask:
            inpaint_chain_idx += 1
            prev_inpaint_len=len(inpaint)
            for subcon in con.split(","):
                subcon=subcon.split("/")[0]
                if subcon[0].isalpha(): # this is a part of the motif because the first element of the contig is the chain letter
                    ref_to_add=[(subcon[0], i) for i in np.arange(int(subcon.split("-")[0][1:]),int(subcon.split("-")[1])+1)]
                    inpaint.extend(ref_to_add)
                    inpaint_hal.extend([(chain_order[inpaint_chain_idx], i) for i in np.arange(inpaint_idx,inpaint_idx+len(ref_to_add))])
                    inpaint_idx += len(ref_to_add)
                    if self.contig_atoms is not None:
                        for k, v in self.contig_atoms.items():
                            chain_residue = (k[0], int(k[1:]))
                            if chain_residue in inpaint:
                                if v in [['all'], ['*']]:
                                    v = self.atom_names(chain_residue[0], chain_residue[1])
                                atomize_resnum2atomnames[chain_residue] = v
                else:
                    inpaint.extend([('_','_')] * int(subcon.split("-")[0]))
                    inpaint_hal.extend([(chain_order[inpaint_chain_idx], i) for i in np.arange(inpaint_idx,inpaint_idx+int(subcon.split("-")[0]))])
                    inpaint_idx += int(subcon.split("-")[0])
            chains.append((inpaint_chain_idx,prev_inpaint_len,len(inpaint),inpaint_idx))
            inpaint_idx = inpaint_idx + 32

        return inpaint, inpaint_hal, atomize_resnum2atomnames

    def get_inpaint_seq_str(self, inpaint_s):
        '''
        function to generate inpaint_str or inpaint_seq masks specific to this contig
        '''
        s_mask = np.copy(self.mask_1d)
        inpaint_s_list = []
        for i in inpaint_s:
            if '-' in i:
                inpaint_s_list.extend([(i[0],p) for p in range(int(i.split("-")[0][1:]), int(i.split("-")[1])+1)])
            else:
                inpaint_s_list.append((i[0],int(i[1:])))
        for res in inpaint_s_list:
            if res in self.ref:
                s_mask[self.ref.index(res)] = False #mask this residue
    
        return np.array(s_mask) 

    def get_idx0(self):
        ref_idx0=[]
        hal_idx0=[]
        ref_idx0_inpaint=[]
        hal_idx0_inpaint=[]

        for idx, val in enumerate(self.ref):
            if val != ('_','_'):
                assert val in self.parsed_pdb['pdb_idx'],f"{val} is not in pdb file!"
                hal_idx0.append(idx)
                ref_idx0.append(self.parsed_pdb['pdb_idx'].index(val))
        for idx, val in enumerate(self.inpaint):
            if val != ('_','_'):
                hal_idx0_inpaint.append(idx)
                ref_idx0_inpaint.append(self.parsed_pdb['pdb_idx'].index(val))

        return ref_idx0, hal_idx0, ref_idx0_inpaint, hal_idx0_inpaint

    def get_mappings(self):
        return dict(
            con_ref_pdb_idx=[i for i in self.inpaint if i[:2] != ('_','_')],
            con_hal_pdb_idx=[self.inpaint_hal[i] for i in range(len(self.inpaint_hal)) if self.inpaint[i][:2] != ("_","_")],
            con_ref_idx0=np.array(self.ref_idx0_inpaint),
            con_hal_idx0=np.array(self.hal_idx0_inpaint),
            inpaint_str=self.inpaint_str,
            inpaint_seq=self.inpaint_seq,
            sampled_mask=self.sampled_mask,
            mask_1d=self.mask_1d,
            atomize_indices2atomname=self.atomize_indices2atomname,
        )

    def res_list_to_mask(self, res_list):
        '''
        Using self.ref as the guide (which refers to the numbering in the input pdb)
        Return a mask of residues specified by res_list. Think ppi hotspot list
        res_list should be comma separated and have chain letters for all residues
        Ligand residues can also be specified with LIGNAME:ATOM. Atom ranges are allowed and are in pdb order. Drop spaces from ligand names

        Args:
            res_list (str): comma separated list to select, must have chain letters for all residues (A5,A6-10,B12 etc). Ligands like LG1:C1-C9

        Returns:
            mask (torch.Tensor[bool]): Which residues were selected
        '''
        mask = torch.zeros((len(self.ref),), dtype=bool)

        ref_stripped = [(x.strip() if isinstance(x, str) else x, y.strip() if isinstance(y, str) else y) for x,y in self.ref]

        for part in res_list.split(','):
            if ':' in part:
                lig_name, atom_part = part.split(':')
                if '-' in atom_part:
                    start,stop = atom_part.split('-')
                else:
                    start = stop = atom_part
                assert (lig_name, start) in ref_stripped, f'Atom {lig_name}:{start} not found. From residue list: {res_list}'
                start_idx = ref_stripped.index((lig_name, start))
                assert (lig_name, stop) in ref_stripped, f'Atom {lig_name}:{stop} not found. From residue list: {res_list}'
                stop_idx = ref_stripped.index((lig_name, stop))
                mask[start_idx:stop_idx+1] = True
            else:
                goal_chain = part[0]
                assert goal_chain.isalpha(), f'Residue list: {res_list} missing chain identifier on this part: {part}'
                if '-' in part[1:]:
                    start,stop = int(part[1:].split('-')[0]),int(part[1:].split('-')[1])
                else:
                    start = stop = int(part[1:])
                for goal_num in range(start, stop+1):
                    found = False
                    for i, (chain,num) in enumerate(self.ref):
                        if chain == goal_chain and num == goal_num:
                            mask[i] = True
                            found = True
                            break
                    assert found, f'Residue {goal_chain}{goal_num} not found. From residue list: {res_list}'
        return mask
