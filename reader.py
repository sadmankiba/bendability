from __future__ import annotations

from constants import CNL, RL, SEQ_LEN, TL, CHRVL, LIBL
from custom_types import YeastChrNum
from util import roman_to_num

import pandas as pd
from Bio import SeqIO
import numpy as np

import math
import inspect
from pathlib import Path

CNL_FILE = '41586_2020_3052_MOESM4_ESM.txt'
RL_FILE = '41586_2020_3052_MOESM6_ESM.txt'
TL_FILE = '41586_2020_3052_MOESM8_ESM.txt'
CHRVL_FILE = '41586_2020_3052_MOESM9_ESM.txt'
LIBL_FILE = '41586_2020_3052_MOESM11_ESM.txt'


class DNASequenceReader:
    """
    Reads and returns processed DNA sequence libraries
    """
    def __init__(self):
        # Get current directory of this module in runtime. With this, we can
        # create correct path even when this module is called from modules in
        # other directories. (e.g. child directory)
        parent_dir = Path(inspect.getabsfile(inspect.currentframe())).parent

        self._bendability_data_dir = f'{parent_dir}/data/input_data/bendability'
        self._nuc_center_file = f'{parent_dir}/data/input_data/nucleosome_position/41586_2012_BFnature11142_MOESM263_ESM.txt'

    def _get_raw_data(self):
        cnl_df_raw = pd.read_table(f'{self._bendability_data_dir}/{CNL_FILE}',
                                   sep='\t')
        rl_df_raw = pd.read_table(f'{self._bendability_data_dir}/{RL_FILE}',
                                  sep='\t')
        tl_df_raw = pd.read_table(f'{self._bendability_data_dir}/{TL_FILE}',
                                  sep='\t')
        chrvl_df_raw = pd.read_table(
            f'{self._bendability_data_dir}/{CHRVL_FILE}', sep='\t')
        libl_df_raw = pd.read_table(
            f'{self._bendability_data_dir}/{LIBL_FILE}', sep='\t')

        return (cnl_df_raw, rl_df_raw, tl_df_raw, chrvl_df_raw, libl_df_raw)

    def _preprocess(self, df: pd.DataFrame):
        df = df[["Sequence #", "Sequence",
                 " C0"]].rename(columns={" C0": "C0"})
        df['Sequence'] = df['Sequence'].str[25:-25]

        return df

    def get_processed_data(self) -> dict[str, pd.DataFrame]:
        """
        Get processed DNA sequence libraries

        returns :
            A dict mapping library names to 
                pandas Dataframe with columns `["Sequence #", "Sequence", "C0"]`
            
        """
        #TODO : Read specific library given as input parameters instead of all
        (cnl_df_raw, rl_df_raw, tl_df_raw, chrvl_df_raw,
         libl_df_raw) = self._get_raw_data()

        cnl_df = self._preprocess(cnl_df_raw)
        rl_df = self._preprocess(rl_df_raw)
        tl_df = self._preprocess(tl_df_raw)
        chrvl_df = self._preprocess(chrvl_df_raw)
        libl_df = self._preprocess(libl_df_raw)

        return {
            CNL: cnl_df,
            RL: rl_df,
            TL: tl_df,
            CHRVL: chrvl_df,
            LIBL: libl_df
        }

    # TODO: nuc_center in separate class
    def read_nuc_center(self) -> pd.DataFrame:
        """
        Read nucleosome center position data. 
        
        Data is provided by paper "A map of nucleosome positions in yeast at base-pair resolution"

        Returns:
            A `pandas.DataFrame` with columns ['Chromosome ID', 'Position', 'NCP score', 'NCP score/noise']
        """
        return pd.read_table(self._nuc_center_file,
                             delim_whitespace=True,
                             header=None,
                             names=[
                                 'Chromosome ID', 'Position', 'NCP score',
                                 'NCP score/noise'
                             ])

    # TODO: read_genome_sequence_of
    def read_yeast_genome(self, chr_num: YeastChrNum) -> pd.DataFrame:
        """
        Read reference sequence of a yeast chromosome. Transforms it into 50-bp
        sequences at 7-bp resolution. 
        
        Returns:
            A pandas DataFrame with columns ['Sequence #', 'Sequence']
        """
        chr = roman_to_num(chr_num)
        assert chr >= 1 and chr <= 16

        # Read file
        genome_file = open(
            "data/input_data/yeast_genome/S288C_reference_sequence_R64-3-1_20210421.fsa"
        )
        fasta_sequences = SeqIO.parse(genome_file, 'fasta')

        # Get sequence of a chromosome
        ref_str = f'ref|NC_00{str(1132 + chr)}|'
        seq = list(filter(lambda fasta: fasta.id == ref_str,
                          fasta_sequences))[0].seq
        genome_file.close()

        # Split into 50-bp sequences
        num_50bp_seqs = math.ceil((len(seq) - SEQ_LEN + 1) / 7)
        seqs_50bp = list(
            map(lambda seq_idx: str(seq[seq_idx * 7:seq_idx * 7 + 50]),
                range(num_50bp_seqs)))

        return pd.DataFrame({
            'Sequence #': np.arange(num_50bp_seqs) + 1,
            'Sequence': seqs_50bp
        })


class GeneReader:
    def read_genes_of(self, chrm_num: YeastChrNum) -> pd.DataFrame:
        genes_file = 'data/input_data/gene/yeast_genes.tsv'
        rename_map = {
            'Gene.length': 'length',
            'Gene.chromosome.primaryIdentifier': 'chrm',
            'Gene.chromosomeLocation.start': 'start',
            'Gene.chromosomeLocation.end': 'end',
            'Gene.chromosomeLocation.strand': 'strand'
        }
        gene_df = pd.read_csv(genes_file, sep='\t')[list(rename_map.keys())]\
            .rename(columns=rename_map)

        return gene_df.loc[gene_df['chrm'] == f'chr{chrm_num}'].drop(columns='chrm')

    def read_transcription_regions_of(self, chrm_num: YeastChrNum) -> pd.DataFrame:
        """
        Read trascripted regions. 

        Besides coding region, a transcripted region also includes untranslated
        regions (UTR) upstream and downstream of it. 

        Returns: 
            A Pandas dataframe with columns ['start', 'end', 'strand']
        """
        gene_utrs_file = 'data/input_data/gene/yeast_gene_utrs.tsv'
        rename_map = {
            'Gene.secondaryIdentifier': 'gene_id',
            'Gene.transcripts.chromosome.primaryIdentifier': 'chrm',
            'Gene.transcripts.UTRs.chromosomeLocation.start': 'utr_start',
            'Gene.transcripts.UTRs.chromosomeLocation.end': 'utr_end',
            'Gene.chromosomeLocation.strand': 'strand',
            # 'Gene.transcripts.UTRs.primaryIdentifier': 'utr_id'
        }
        # Read file and rename columns
        tr_df = pd.read_csv(gene_utrs_file, sep='\t')[list(rename_map.keys())]\
            .rename(columns=rename_map)

        # Select genes of particular chromosome 
        tr_df = tr_df.loc[tr_df['chrm'] == f'chr{chrm_num}'].drop(columns='chrm')

        aggregation = {
            'starts_min': pd.NamedAgg(column='utr_start', aggfunc='min'),
            'starts_max': pd.NamedAgg(column='utr_start', aggfunc='max'),
            'ends_min': pd.NamedAgg(column='utr_end', aggfunc='min'),
            'ends_max': pd.NamedAgg(column='utr_end', aggfunc='max')
        }

        # Each gene has 2 UTRs. 
        # Group by gene id and create 4 columns for 4 start and end positions of 5' and 3' utr
        agg_tr_df = tr_df.groupby(['gene_id', 'strand']).agg(**aggregation)
        
        # Find min and max of 4 start and end positions of 5' and 3' utr of each gene
        agg_tr_df['start'] = agg_tr_df.apply(lambda gene: min(gene[list(aggregation.keys())]), axis=1)
        agg_tr_df['end'] = agg_tr_df.apply(lambda gene: max(gene[list(aggregation.keys())]), axis=1)
        
        # Convert multi-index to column 
        agg_tr_df = agg_tr_df.reset_index()

        # Drop unnecessary columns 
        agg_tr_df = agg_tr_df.drop(columns=list(aggregation.keys()) + ['gene_id'])
        assert len(agg_tr_df) == len(tr_df) / 2
        
        return agg_tr_df 

        