# Explainer and definitions -- Multiple variables

The first explainer introduced the philosophy of the analysis and the most important definitions. You might have noted that it assumed **one variable**. Generally, we will have multiple variables, which can be *measured* at one or more points. Also generally, two different variables might be measured at the same points, never at the same point, or somewhere in between. I call **overlap** the ratio of overlapping measurements between two variables, and this parameter allows me to expand the analysis of the different data structures to synthetic datasets with multiple variables.

To be more specific, the overlap between two variables is measured across all shared dimensions. For example, if we assume that two instruments measure variable Var1 and variable Var2 at a given location and time (3 dimensions: latitude, longitude, time), then overlap = 1 if both variables Var1 and Var2 are measured at the same locations and same times. If the instrument that measures Var2 has half the frequency of the instrument that measures Var1, then it will be overlap = 0.5. Overlap = 0.5 also if, for the same measuring frequency, one instrument stopped recording after half the measurements. If Var2 has only 2 dimensions (e.g. Var2 is the ID of each location at a given (latitude,longitude) so it's measured once and the time of the measurement is not recorded because not relevant)n then overlap is measured along latitude and longitude. In this case, overlap = 0.5 means that only half the (latitude,longitude) pairs that contain Var1 measurements also contain Var2 values.

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
