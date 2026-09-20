# Explainer and definitions -- Multiple variables

The first explainer introduced the philosophy of the analysis and the most important definitions. You might have noted that it assumed **one variable**. Generally, we will have multiple variables, which can be *measured* at one or more points. Also generally, two different variables might be measured at the same points, never at the same point, or somewhere in between. I call **overlap** the ratio of overlapping measurements between two variables, and this parameter allows me to expand the analysis of the different data structures to synthetic datasets with multiple variables.

To be more specific, the overlap between two variables is measured across all shared dimensions. For example, if we assume that two instruments measure variable Var1 and variable Var2 at a given location and time (3 dimensions: latitude, longitude, time), then overlap = 1 if both variables Var1 and Var2 are measured at the same locations and same times. If the instrument that measures Var2 has half the frequency of the instrument that measures Var1, then it will be overlap = 0.5. Overlap = 0.5 also if, for the same measuring frequency, one instrument stopped recording after half the measurements. If Var2 has only 2 dimensions (e.g. Var2 is the ID of each location at a given (latitude,longitude) so it's measured once and the time of the measurement is not recorded because not relevant)n then overlap is measured along latitude and longitude. In this case, overlap = 0.5 means that only half the (latitude,longitude) pairs that contain Var1 measurements also contain Var2 values.

#### Overlap, as a formula

Writing `S_i` for the set of sites variable *i* occupies and `proj` for the projection onto the
dimensions two variables share, the overlap of variable *i* against the reference variable
`var0` is

```
    O_i = |proj(S_0) & proj(S_i)| / |proj(S_0)|
```

the share of the reference variable's sites that also carry variable *i*. This is the quantity
the `overlap` parameter targets. When both variables vary along the same dimensions the
projection does nothing and it reduces to `|S_0 & S_i| / |S_0|`.

The reverse ratio is also reported, because either can be the one a reader expects:

```
    O'_i = |proj(S_0) & proj(S_i)| / |proj(S_i)|
```

They are related by `O'_i = O_i * |proj(S_0)| / |proj(S_i)|`. Note that this is the ratio of the
**projected** set sizes, not of the observation counts: projection collapses many reference
sites onto one cell, so `|proj(S_0)|` is generally smaller than `S_0`.

Two consequences worth knowing before choosing a target:

* `O_i` cannot exceed `|proj(S_i)| / |proj(S_0)|`, since the intersection cannot be larger than
  either set. If variable *i* has half as many sites as the reference, its overlap cannot exceed
  0.5 — which is exactly the instrument reading above, where halving the measuring frequency
  halves the overlap.
* For a variable that drops a dimension, a cell of the shared space is free of the reference
  only if the reference misses it at *every* dropped coordinate. That is rare, so such a
  variable has little room to sit off the reference and its overlap is pushed towards its own
  density. Where the requested target cannot be reached the generator says so and the density
  takes precedence.

### Tabular vs array structures
In the tabular format, every column must have a value for each row, so if the variable Var1 has three dimensions, and the variable Var2 has two dimensions, still we need to store some information about the third dimension wherever we measure Var2, even if no measurement of Var1 is present.

Scenarios for Var1 with dimensions X, Y, Z and Var2 with dimensions X and Y.

1) Var1 measured at (X0, Y0, Z0), Var2 not measured at (X0, Y0)

X | Y | Z | Var1 | Var2
X0 | Y0 | Z0 | Var1_000 | nan |

1b) Var1 measured at multiple Z at (X0, Y0), Var2 not measured at (X0, Y0)

X | Y | Z | Var1 | Var2
X0 | Y0 | Z0 | Var1_000 | nan |
X0 | Y0 | Z1 | Var1_001 | nan |
X0 | Y0 | Z2 | Var1_002 | nan |
X0 | Y0 | Z3 | Var1_003 | nan |

2) Var1 measured at (X0, Y0, Z0), Var2 measured at (X0, Y0)

X | Y | Z | Var1 | Var2
X0 | Y0 | Z0 | Var1_000 | Var2_00 |

3) Var1 not measured at any (X0, Y0, Z), Var2 measured at (X0, Y0)

X | Y | Z | Var1 | Var2
X0 | Y0 | nan | nan | Var2_00 |
