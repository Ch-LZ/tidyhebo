"""Regressions for the extracted population initializer."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tidyhebo.layered.config import InitializationConfig
from tidyhebo.layered.population_initialization import initialize_population

class _Fake:
    def __init__(self, q):
        self.q=q
        self.calls=[]
        self.level=SimpleNamespace(population=7)
        self.directions=np.array([[.1,.6,.3],[.3,.3,.4],[.8,.1,.1]])
    def sobol(self,n):
        self.calls.append(('sobol',n))
        return torch.rand(n,self.q,2, dtype=torch.double)
    def clean(self,x):
        self.calls.append(('clean',len(x)))
        return x[x[:,0,0]>=.15]
    def _optimize_targets(self,raw,rays,objectives,restarts,steps):
        self.calls.append(('optimize',tuple(objectives),tuple(map(tuple,rays)),restarts,steps))
        ids=np.random.randint(len(raw),size=len(rays))
        return raw[torch.as_tensor(ids)]+torch.rand(len(rays),self.q,2)*.001

# This is a reference copy from the old LayeredSearch._initialize.
def _original(self, config, supplied=None):
    n=self.level.population
    raw=self.clean(self.sobol(max(n,config.restarts*8)))
    if not len(raw):raise RuntimeError('No feasible Sobol seeds')
    parts=[]
    rays,objectives=[],[]
    if config.anchors:
        rays.extend([self.directions[0]]*3)
        objectives.extend(range(3))
    if config.directions:
        count=len(self.directions) if config.directions==-1 else min(config.directions,len(self.directions))
        rays.extend(self.directions[i] for i in np.linspace(0,len(self.directions)-1,count,dtype=int))
        objectives.extend([-1]*count)
    if rays:parts.append(self._optimize_targets(raw,np.asarray(rays),objectives,config.restarts,config.local_steps))
    if supplied is not None and len(supplied):parts.append(supplied)
    seeds=self.clean(torch.cat(parts))[:n] if parts else raw[:0]
    for _ in range(40):
        if len(seeds)>=n:break
        seeds=self.clean(torch.cat((seeds,self.sobol(max(n,n-len(seeds))))))[:n]
    if len(seeds)<n:raise RuntimeError(f'Only {len(seeds)} feasible batches for population={n}')
    return seeds

def _rng():return np.random.get_state(),torch.random.get_rng_state().clone()

def _eq_rng(a,b):
    assert a[0][0]==b[0][0]
    np.testing.assert_array_equal(a[0][1],b[0][1]);assert a[0][2:]==b[0][2:]
    assert torch.equal(a[1],b[1])

@pytest.mark.parametrize('q,anchors,directions,supplied_n',[
    (1,True,2,0),(2,True,-1,2),(4,False,2,2),(4,False,0,0),(2,True,0,1),
])
def test_initializer_replays_old_calls_and_rng(q,anchors,directions,supplied_n):
    conf=replace(InitializationConfig(),anchors=anchors,directions=directions,restarts=2,local_steps=3)
    supplied=torch.full((supplied_n,q,2),.55,dtype=torch.double) if supplied_n else None
    np.random.seed(2026);torch.manual_seed(2026)
    old=_Fake(q);expected=_original(old,conf,supplied);before=_rng()
    np.random.seed(2026);torch.manual_seed(2026)
    new=_Fake(q)
    actual=initialize_population(conf,population=new.level.population,reference_directions=new.directions,
        sobol=new.sobol,clean=new.clean,optimize_targets=new._optimize_targets,supplied=supplied)
    assert torch.equal(actual,expected)
    assert old.calls==new.calls
    _eq_rng(before,_rng())
