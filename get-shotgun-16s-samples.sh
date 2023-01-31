#!/bin/sh

# Retrive shotgun and 16s samples from redbiom
# redbiom summarize contexts | grep wolka                                     # this retrieve all contexts related to shotgun
wgsctx=Woltka-per-genome-WoLr2-3ab352                                       # save context string as bash variable
# redbiom summarize contexts | grep -i deblur | grep 150nt | grep -i v4       # retrives all contexts related to 16s data with 150 read lengths targeting the v4 region of 16s rna
ctx16s=Deblur_2021.09-Illumina-16S-V4-150nt-ac8c0b                       # save context as bash variable
redbiom fetch samples-contained --context $wgsctx > samples/wgs.samples             # get sample names of shotgun samples
redbiom fetch samples-contained --context $ctx16s > samples/16s.samples             # get sample names of 16s samples

# Find the paired samples from the previous step
python samples/extract-paired-samples.py

# remove 16s.samples and wgs.samples files (optional)
# rm samples/16s.samples
# rm samples/wgs.samples

# get paired biom tables (this will take a while)
redbiom fetch samples --from samples/common.samples --context $wgsctx --output samples/wgs.biom --resolve-ambiguities 'most-reads' # get shotgun biom tables
redbiom fetch samples --from samples/common.samples --context $ctx16s --output samples/16s.biom --resolve-ambiguities 'most-reads' # get 16s biom tables

# get sample-metadata
redbiom fetch sample-metadata --output samples/wgs-metadata.txt --from samples/common.samples --context $wgsctx --resolve-ambiguities
redbiom fetch sample-metadata --output samples/16s-metadata.txt --from samples/common.samples --context $ctx16s --resolve-ambiguities

# # extract paired feature vectors
# python samples/generate-training-data.py
