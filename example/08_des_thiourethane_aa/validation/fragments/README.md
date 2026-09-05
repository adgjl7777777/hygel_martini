# DFT fragment force fields, for the pair benchmark only

`F1.itp` / `F2.itp` / `F3.itp` are OPLS-AA parameters for the three model
compounds the DFT side used, obtained the same way every template in this
example was: the fragment geometry taken **out of the DFT complex** (Cl-
removed) was submitted to LigParGen with `checkopt=3` and
`chargetype=cm1abcc`, i.e. 1.14*CM1A-LBCC, and the returned ITP/GRO kept
verbatim.

| file | fragment | formula | from |
|---|---|---|---|
| `F1` | methyl 3-mercaptopropionate (thiol) | C4H8O2S | `C1_thiol_Cl/opt.xyz` |
| `F2` | iPrO-C(=O)-NH-p-tolyl (urethane) | C11H15NO2 | `C2_urethane_Cl/opt.xyz` |
| `F3` | MeS-C(=O)-NH-p-tolyl (thiourethane) | C9H11NOS | `C3_thiouret_Cl/opt.xyz` |

F3x needs no file here: that fragment **is** `parameterization/raw/LNK.itp`,
the model compound the builder's own crossing parameters came from, atom for
atom (C12H15NO3S, 32 atoms).

These exist so the benchmark is reproducible without a network call. They are
**not** part of any simulation topology and must never be included in one --
they are duplicates of chemistry that HEXU/STR/LNK already carry, under
LigParGen's generic `UNK` moleculetype, and would collide.

The `*_frag.pdb` files are the exact submissions, so the round trip can be
repeated or checked.
