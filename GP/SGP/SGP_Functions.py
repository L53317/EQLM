# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2020 University of Strathclyde and Author ------
# ---------------- Author: Francesco Marchetti ------------------------
# ----------- e-mail: francesco.marchetti@strath.ac.uk ----------------

# Alternatively, the contents of this file may be used under the terms
# of the GNU General Public License Version 3.0, as described below:

# This file is free software: you may copy, redistribute and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3.0 of the License, or (at your
# option) any later version.

# This file is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import sys
import os
import numpy as np
from copy import copy
from deap import tools
from deap.algorithms import varOr
from operator import attrgetter
import random
from functools import partial
gpfun_path = os.path.join(os.path.dirname( __file__ ), '../IGP/Examples', '..')
sys.path.append(gpfun_path)
import IGP_Functions as funs
try:
    import models_FESTIP as mods
except ModuleNotFoundError:
    mods = None

def eaMuPlusLambdaTolSimple(population, toolbox, mu, lambda_, ngen, cxpb, mutpb, pset, creator, stats=None,
                            halloffame=None, verbose=__debug__, **kwargs):
    """Modification of eaMuPlusLambda function from DEAP library, used by SGP. Modifications include:
        - use of tolerance value for the first fitness function below which the evolution is stopped
        - implemented tolerance stopping criteria
        - use of custom POP class to keep track of the evolution of the population

        Original description:

        This is the :math:`(\mu + \lambda)` evolutionary algorithm.
        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param mu: The number of individuals to select for the next generation.
        :param lambda\_: The number of children to produce at each generation.
        :param cxpb: The probability that an offspring is produced by crossover.
        :param mutpb: The probability that an offspring is produced by mutation.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                      inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                           contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                  evolution.
        The algorithm takes in a population and evolves it in place using the
        :func:`varOr` function. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evaluations for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *cxpb* and *mutpb* arguments are passed to the
        :func:`varOr` function. The pseudocode goes as follow ::
            evaluate(population)
            for g in range(ngen):
                offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
                evaluate(offspring)
                population = select(population + offspring, mu)
        First, the individuals having an invalid fitness are evaluated. Second,
        the evolutionary loop begins by producing *lambda_* offspring from the
        population, the offspring are generated by the :func:`varOr` function. The
        offspring are then evaluated and the next generation population is
        selected from both the offspring **and** the population. Finally, when
        *ngen* generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.
        This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox. This algorithm uses the :func:`varOr`
        variation.
        """

    if 'v_wind' in kwargs:
        init_wind = copy(kwargs['v_wind'])
        init_delta = copy(kwargs['deltaH'])

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    data = np.array(['Min length', 'Max length', 'Entropy', 'Distribution'])
    all_lengths = []
    pop = funs.POP(population, creator)
    data, all_lengths = pop.save_stats(data, all_lengths)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(partial(toolbox.evaluate, pset=pset, kwargs=kwargs), invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None and kwargs['mod_hof'] is True:
        halloffame.update(population, for_feasible=True)
    if halloffame is not None and kwargs['mod_hof'] is False:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    min_fit = np.array(logbook.chapters["fitness"].select("min"))

    if kwargs['fit_tol'] is None and kwargs['check'] is True:
        success = mods.check_success(toolbox, halloffame, **kwargs)
    elif kwargs['fit_tol'] is not None and kwargs['check'] is False:
        if min_fit[-1][0] < kwargs['fit_tol'] and min_fit[-1][-1] == 0:
            success = True
        else:
            success = False
    elif kwargs['fit_tol'] is None and kwargs['check'] is False:
        success = False

    # Begin the generational process
    gen = 1
    while gen < ngen + 1 and not success:
        # Vary the population

        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        if 'v_wind' in kwargs:
            eps = 0.1
            v_wind = random.uniform(init_wind * (1 - eps), init_wind * (1 + eps))
            deltaT = random.uniform(init_delta * (1 - eps), init_delta * (1 + eps))
            print("New point: Height start {} km, Wind speed {} m/s, Range gust {} km".format(kwargs['height_start'] / 1000, v_wind,
                                                                                              deltaT / 1000))

        # Evaluate the individuals with an invalid fitness

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(partial(toolbox.evaluate, pset=pset, kwargs=kwargs), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None and kwargs['mod_hof'] is True:
            halloffame.update(offspring, for_feasible=True)
        if halloffame is not None and kwargs['mod_hof'] is False:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        pop = funs.POP(population, creator)
        data, all_lengths = pop.save_stats(data, all_lengths)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)
        gen += 1

    return population, logbook, data, all_lengths


def xselDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first):
    """
    From [2]
    """
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            lind1 = sum([len(gpt) for gpt in ind1])
            lind2 = sum([len(gpt) for gpt in ind2])
            if lind1 > lind2:
                ind1, ind2 = ind2, ind1
            elif lind1 == lind2:
                # random selection in case of a tie
                prob = 0.5

            # Since size1 <= size2 then ind1 is selected
            # with a probability prob
            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter("fitness")))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)