# Copyright (C) 2012 Robert Lanfear and Brett Calcott
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details. You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# PartitionFinder also includes the PhyML program, the RAxML program, and the
# PyParsing library, all of which are protected by their own licenses and
# conditions, using PartitionFinder implies that you agree with those licences
# and conditions as well.
import os
from . import logtools
log = logtools.get_logger()

import hashlib
import pickle as pickle
from .util import get_aic, get_aicc, get_bic
from scipy.stats import chi2 
from .util import PartitionFinderError

import os
import numpy

from .alignment import Alignment, SubsetAlignment
from .util import (ParseError, PartitionFinderError, remove_runID_files, get_aic, get_aicc,
                  get_bic)


FRESH, PREPARED, DONE = list(range(3))

class AnalysisError(PartitionFinderError):
    pass

class SubsetError(PartitionFinderError):
    pass


def count_subsets():
    return len(Subset._cache)


def clear_subsets():
    Subset._cache.clear()

def subset_unique_name(columns):
    """Return a unique string based on the subsets columns (which are unique)"""

    # Use pickle to give us a string from the object
    pickled_columns = pickle.dumps(columns, -1)

    # Now get an md5 hash from this. There is some vanishingly small chance that
    # we'll get the same thing. Google "MD5 Hash Collision"
    return hashlib.md5(pickled_columns).hexdigest()


class Subset(object):
    """Contains a set of columns in the Alignment
    """
    _cache = {}

    def __new__(cls, cfg, column_set, name=None, description=None):
        """Returns the identical subset if the columns are identical.

        This is basically a pythonized factory. See here:
        http://codesnipers.com/?q=python-flyweights
        """
        columns = list(column_set)
        columns.sort()
        subset_id = subset_unique_name(columns)
        obj = Subset._cache.get(subset_id, None)
        if not obj:
            obj = object.__new__(cls)
            Subset._cache[subset_id] = obj
            obj.init(subset_id, cfg, column_set, columns)

        return obj

    def init(self, subset_id, cfg, column_set, columns):
        self.subset_id = subset_id
        self.cfg = cfg
        self.column_set = column_set
        self.columns = columns
        self.status = FRESH
        self.names = []
        self.description = []

        # We put all results into this array, which is sized to the number of
        # models that we are analysing
        self.result_array = numpy.zeros(
            cfg.model_count, cfg.data_layout.data_type)

        # This points to the current empty array entry that we will fill up
        # next. When we're done it will equal the size of the array (and
        # accessing will cause an error)
        self.result_current = 0

        # This will get set to the best entry, once we've done all the analysis
        self.result_best = None

        self.models_not_done = set(cfg.models)

        self.fabricated = False
        self.dont_split = False
        self.analysis_error = None
        self.centroid = None

        # Site likelihoods calculated using GTR+G from the
        # processor.gen_per_site_stats()
        self.site_lnls_GTRG = []

        self.alignment_path = None
        log.debug("Created %s" % self)

    def add_description(self, names, description):
        """User created subsets can get some extra info"""
        self.names = names
        self.description = description

    @property
    def name(self):
        try:
            return ", ".join(self.names)
        except:
            return "NA"

    def get_site_description(self, use_commas):
        try:
            s = []
            for desc in self.description:
                step = desc[2]
                if step == 1:
                    text = "%s-%s" % (desc[0], desc[1])
                else:
                    text = "%s-%s\\%s" % tuple(desc)
                s.append(text)

            if use_commas:
                site_description = ', '.join(s)
            else:
                site_description = ' '.join(s)
            return site_description
        except:
            return ', '.join(map(str, self.columns))

    @property
    def site_description(self):
        return self.get_site_description(use_commas = True)

    @property
    def site_description_no_commas(self):
        return self.get_site_description(False)

    def __repr__(self):
        return "Subset(%s..)" % self.name[:5]

    def load_results(self, cfg):
        matching = cfg.database.get_results_for_subset(self)
        # We might get models that we don't want, so we need to filter them
        for i, mod in enumerate(matching['model_id']):
            if mod in self.models_not_done:
                self.result_array[self.result_current] = matching[i]
                self.result_current += 1
                self.models_not_done.remove(mod)

    def add_result(self, cfg, model, result):
        """
        We get the result class from raxml or phyml. We need to transform this
        into a numpy record, and then store it locally, and in the database
        """
        K = float(cfg.processor.models.get_num_params(model))
        n = float(len(self.column_set))
        lnL = float(result.lnl)
        aic = get_aic(lnL, K)
        bic = get_bic(lnL, K, n)
        aicc = get_aicc(lnL, K, n)

        result.subset_id = self.subset_id
        result.model_id = model
        result.params = K
        result.aic = aic
        result.aicc = aicc
        result.bic = bic

        # Now assign the data record in the result into the current model
        # result and save it to the database
        self.result_array[self.result_current] = result._data
        cfg.database.save_result(self, self.result_current)
        self.result_current += 1

        log.debug("Added model to subset. Model: %s, params: %d, sites:%d, lnL:%.2f, site_rate %f"
                  % (model, K, n, lnL, result.site_rate))

    def model_selection(self, cfg):
        # We want the index of the smallest value
        method = cfg.model_selection
        self.result_best = numpy.argmin(self.result_array[method])
        best = self.result_array[self.result_best]

        # TODO: this is crappy. Anyone who wants this stuff should just access
        # the entire "best" item
        self.best_info_score = best[method]
        self.best_lnl = best['lnl']
        self.best_model = best['model_id']
        self.best_site_rate = best['site_rate']
        self.best_params = best['params']
        self.best_alpha = best['alpha']
        self.best_freqs = best['freqs']
        self.best_rates = best['rates']

        log.debug("Best model for this subset: %s \n" 
                  "lnL: %s\n" 
                  "site_rate: %s\n" 
                  "params: %s\n" 
                  "alpha: %s\n" 
                  "freqs: %s\n" 
                  "rates: %s\n"
                  % (self.best_model, str(self.best_lnl),
                    str(self.best_site_rate), str(self.best_params),
                    str(self.best_alpha), str(self.best_freqs),
                    str(self.best_rates)))

    def get_param_values(self):
        param_values = {
            "rate": self.best_site_rate,
            "model": self.best_rates,
            "alpha": self.best_alpha,
            "freqs": self.best_freqs,
        }
        return param_values



    def finalise(self, cfg):

        log.debug("Finalising subset %s", self.subset_id)

        log.debug("models not done: %s", self.models_not_done)

        if self.models_not_done:
            return False

        # We might already have done everything
        if self.status == DONE:
            return True

        self.model_selection(cfg)

        # Do all the final cleanup
        if cfg.save_phylofiles:
            # Write out a summary of the subsets
            cfg.reporter.write_subset_summary(self)
        else:
            # Otherwise, clean up files generated by the programs as well
            if self.alignment_path:
                remove_runID_files(self.alignment_path)

        self.models_to_process = []
        self.status = DONE
        cfg.progress.subset_done(self)


        return True

    def prepare(self, cfg, alignment):
        """Get everything ready for running the analysis
        """
        log.debug("Preparing to analyse subset %s", self.name)
        cfg.progress.subset_begin(self)

        # Load the cached results
        self.load_results(cfg)
        if self.finalise(cfg):
            return

        # Make an Alignment from the source, using this subset
        self.make_alignment(cfg, alignment)
        self.models_to_process = list(self.models_not_done)
        # Now order them by difficulty
        self.models_to_process.sort(
            key=cfg.processor.models.get_model_difficulty,
            reverse=True)

        self.status = PREPARED

    def parse_results(self, cfg):
        """Read in the results and parse them"""
        for m in list(self.models_not_done):
            self.parse_model_result(cfg, m)

    def fabricate_model_result(self, cfg, model):
        self.fabricated = True
        self.dont_split = True
        self.models_not_done.remove(model)

    def parse_model_result(self, cfg, model):
        pth, tree_path = cfg.processor.make_output_path(
            self.alignment_path, model)

        if not os.path.exists(pth):
            # If it ain't there, we can't do it
            return

        output = open(pth, 'rb').read()
        try:
            result = cfg.processor.parse(output, cfg)
            self.add_result(cfg, model, result)
            # Remove the current model from remaining ones
            self.models_not_done.remove(model)

            if not cfg.save_phylofiles:
                # We remove all files that have the specified RUN ID
                cfg.processor.remove_files(self.alignment_path, model)

        except ParseError:
            # If we're loading old files, this is fine
            if self.status == FRESH:
                log.warning("Failed loading parse output from %s."
                            "Output maybe corrupted. I'll run it again.",
                            pth)
                cfg.processor.remove_files(self.alignment_path, model)
            else:
                # But if we're prepared, then we've just run this. And we're
                # screwed. Reraise the message
                log.error(
                    "Failed to run models %s; not sure why",
                    ", ".join(list(self.models_not_done)))
                raise

    def add_per_site_statistics(self, per_site_stats):
        self.site_lnls = per_site_stats

    def fabricate_result(self, cfg, model):
        '''If the subset fails to be analyzed, we throw some "fabricated"
        results'''
        processor = cfg.processor
        self.fabricated = True

        lnl = sum(self.site_lnls_GTRG)
        result = processor.fabricate(lnl)

        self.add_result(cfg, model, result)
        self.best_params = cfg.processor.models.get_num_params(model)
        self.best_lnl = result.lnl
        self.models_not_done.remove(model)

    def add_centroid(self, centroid):
        self.centroid = centroid

    FORCE_RESTART_MESSAGE = """
    It looks like you have changed one or more of the data_blocks in the
    configuration file, so the new subset alignments don't match the ones
    stored for this analysis.  You'll need to run the program with
    --force-restart
    """

    def make_alignment(self, cfg, alignment):
        # Make an Alignment from the source, using this subset
        sub_alignment = SubsetAlignment(alignment, self)

        sub_path = os.path.join(cfg.phylofiles_path, self.subset_id + '.phy')
        # Add it into the sub, so we keep it around
        self.alignment_path = sub_path

        # Maybe it is there already?
        if os.path.exists(sub_path):
            log.debug("Found existing alignment file %s" % sub_path)
            old_align = Alignment()
            old_align.read(sub_path)

            # It had better be the same!
            if not old_align.same_as(sub_alignment):
                log.error(self.FORCE_RESTART_MESSAGE)
                raise SubsetError
        else:
            # We need to write it
            sub_alignment.write(sub_path)

    @property
    def is_done(self):
        return self.status == DONE

    @property
    def is_prepared(self):
        return self.status == PREPARED

    @property
    def is_fresh(self):
        return self.status == FRESH



def columnset_to_string(colset):
    s = list(colset)
    s.sort()
    # Add one, cos we converted to zero base...
    return ', '.join([str(x+1) for x in s])

def merge_fabricated_subsets(subset_list):
    '''Allows the merging of fabricated subsets and the preservation of their
    centroids and lnls'''
    columns = set()
    lnl = 0
    centroid = []

    # Figure out how many dimensions the centroid is
    centroid_dim = len(subset_list[0].centroid)
    for i in range(centroid_dim):
        centroid.append(0)

    for sub in subset_list:
        columns |= sub.column_set
        lnl += sub.best_lnl
        number = 0
        for observation in centroid:
            observation += sub.centroid[number]
            number += 1

    # Now just take the average of each centroid to be the centroid of the new
    # subset
    centroid = [x/len(subset_list) for x in centroid]

    new_sub = subset.Subset(sub.cfg, columns)

    # Add the centroid and sum of the lnls to the subset. TODO: create
    # functions to update these variables in the subset rather than messing
    # with them directly
    new_sub.centroid = centroid
    new_sub.lnl = lnl
    return new_sub


def merge_subsets(subset_list):
    """Take a set of subsets and merge them together"""
    columns = set()

    # We just need the columns
    names = []
    descriptions = []
    for sub in subset_list:
        columns |= sub.column_set
        descriptions.extend(sub.description)
        names.extend(sub.names)

    newsub = subset.Subset(sub.cfg, columns)
    # Only add the description if it isn't there (we might get back a cache
    # hit)
    if not newsub.names:
        newsub.add_description(names, descriptions)

    return newsub

def subsets_overlap(subset_list):
    columns = set()
    overlapping = []

    for sub in subset_list:
        # If the intersection is non-empty...
        ov = list(sub.column_set & columns)
        if ov:
            overlapping.append(ov)
        columns |= sub.column_set

    return ov

def check_against_alignment(full_subset, alignment, the_config):
    """Check the subset definition against the alignment"""

    alignment_set = set(range(0, alignment.sequence_length))
    leftout = alignment_set - full_subset.column_set
    if leftout:
        log.warning(
            "These columns are missing from the block definitions: %s",
            columnset_to_string(leftout))
        if the_config.no_ml_tree == False:
            log.error(
                "You cannot estimate a Maximum Likelihood (ML) starting tree"
                " (the default behaviour) when you have columns missing from"
                " your data block definitions, because the method we use "
                "to estimate the ML tree requires all sites in the alignment"
                " to be assigned to a data block. We recommend that you "
                " either remove the sites you don't want from your alignment"
                " or (if possible) include the missing sites in appropriate"
                " data blocks. Failing that, you can use the --no-ml-tree "
                " command line option. In this case, a NJ (PhyML) or MP"
                "(RaxML) starting tree will be estimated for your analysis. "
            )
            raise AnalysisError


def split_subset(a_subset, cluster_list):
    """Takes a subset and splits it according to a cluster list,
     then returns the subsets resulting from the split"""
    # Take each site from the first list and add it to a new
    subset_list = a_subset.columns
    subset_columns = []
    list_of_subsets = []
    for cluster in cluster_list:
        list_of_sites = []
        for site in cluster:
            list_of_sites.append(subset_list[site - 1])
        subset_columns.append(set(list_of_sites))

    tracker = 0
    for column_set in subset_columns:
        new_subset = subset.Subset(a_subset.cfg, column_set)
        list_of_subsets.append(new_subset)
        tracker += 1

    return list_of_subsets

def subset_list_score(list_of_subsets, the_config, alignment):
    """Takes a list of subsets and return the aic, aicc, or bic score"""

    lnL, sum_k, subs_len = subset_list_stats(list_of_subsets, the_config, alignment)

    if the_config.model_selection == 'aic':
        return get_aic(lnL, sum_k)
    elif the_config.model_selection == 'aicc':
        return get_aicc(lnL, sum_k, subs_len)
    elif the_config.model_selection == 'bic':
        return get_bic(lnL, sum_k, subs_len)


def subset_list_stats(list_of_subsets, the_config, alignment):
    """Takes a list of subsets and returns the lnL and the number of params"""
    sum_subset_k = 0
    lnL = 0
    subs_len = 0
    for sub in list_of_subsets:
        sum_subset_k += sub.best_params
        lnL += sub.best_lnl
        subs_len += len(sub.columns)
    # Grab the number of species so we know how many params there are
    num_taxa = len(alignment.species)
    # Linked brlens - only one extra parameter per subset
    if the_config.branchlengths == 'linked':
        sum_k = sum_subset_k + (len(list_of_subsets) - 1) + (
            (2 * num_taxa) - 3)
        log.debug("Total parameters from brlens: %d" %
                  ((2 * num_taxa) - 3))
        log.debug("Parameters from subset multipliers: %d" %
                  (len(list_of_subsets) -1))

    # Unlinked brlens - every subset has its own set of brlens
    elif the_config.branchlengths == 'unlinked':
        sum_k = sum_subset_k + (len(list_of_subsets) * (
            (2 * num_taxa) - 3))
        log.debug("Total parameters from brlens: %d" % ((
            2 * num_taxa) - 3) * (len(list_of_subsets)))

    log.debug("Grand_total_parameters: %d", sum_k)

    return lnL, sum_k, subs_len


def subset_list_score_diff(list1, list2, the_config, alignment):
    """Take two lists of subsets and return the score diff as list1 - list2"""
    list1_score = subset_list_score(list1, the_config, alignment)
    list2_score = subset_list_score(list2, the_config, alignment)

    score_diff = list1_score - list2_score

    return score_diff