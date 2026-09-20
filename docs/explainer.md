# Explainer and definitions

## Data points: sparse vs dense, irregular vs gridded, tabular vs array

A set of data points generally lies on a spectrum between irregular and regular. While I have not defined yet what regular mean in the present context, when you read the previous sentence your mind probably used one of three images to picture the two extremes: sparse vs dense (a cloud with many points vs an almost empty region), irregular vs gridded (few spread out points vs several regularly spaced points), or tabular vs array (a list of records vs points identified by their coordinates).

While it's easy to see similarities among sparse/irregular/tabular and dense/gridded/array, some differences exist and so I'll define them in the context of this work for the sake of a deeper understanding of what I set out to do with it.

### Irregular vs gridded

If a set of points is spaced such that we can identify a set of dimensions whose coordinates are regularly spaced, then I call the dataset **gridded**. Example: a set of 9 points on a 3x3 grid with latitude and longitude coordinates, and the points are 1 degree latitude and longitude apart. [TD ADD FIGURE] In this case, I call the data *purely gridded* because the data occupies all available sites (i.e. longitude-latitude pairs).

The opposite end of purely gridded is *purely irregular*: in this case we cannot identify a set of dimensions whose coordinates are regularly spaced where the points are located. For example, 3 points on a 2D space and whose coordinates are all different^1 [TD ADD FIGURE]

Note that these definitions are independent of the nature of the dimensions (spatial, time, indices...)

Anything in between has some degree of both, so I refer to it as gridded or irregular depending on what extreme they look closest to to me, purely based on my gut feeling. This is heavily based on the sparsity of the data, so let's talk about it next.

^1 to be rigorous the coordinates of the occupied sites would also have to be incommensurable or one cangenerate a grid with empty sites that satisfies the purely gridded definition, but because of some later assumptions on data sparsity I stick to this less rigorous definition that makes communication easier.

### Sparse vs dense

I define sparsity as the ratio of sites not occupied by data points. By contrast, density (or occupancy) is the ratio of sites occupied by data points. Note that sparsity+density = 1.

Unambiguously, when all sites are occupied (e.g. TD ADD FIGURE) density = 1 and sparsity = 0.

Sparsity = 1 (density = 0) also unambiguosly identifies the case where all sites are vacant, but it's pretty useless in practice (I want to investigate different storage techniques, and I need data to store).

The most interesting largest sparsity (smallest density) value is the one such that all available coordinates are used at least once (on a square grid like the one below this works out to once and only once; see 'Minimum density, on any grid' further down for why the general condition has to be 'at least'). For example: 3 points on a 3x3 grid, such that no two points share the same coordinates [TD ADD FIGURE]. P1 is located at (lon1,lat1); P2 at (lon3,lat2); P3 at (lon2,lat3). In this case sparsity = 6/9 = 2/3 (density = 1/3). Note that this is the largest sparsity possible for this 3x3 grid: adding any point would lower it (e.g. TD ADD PANEL TO FIGURE), and removing any point would make 2 coordinates pointless^2 (e.g. TD ADD PANEL TO FIGURE), because their knowledge does not add anything, and I don't want to store data that is not required to describe my data points.

This is why I only consider cases where all coordinates are occupied at least once^3. I try to refer to cases like [TD ADD FIGURE] as 'least dense' or 'most sparse', but I might accidentally call them 'purely irregular' or 'irregular', as they are the most irregular I consider in this work.

#### Minimum density, on any grid

On the 3x3 grid above, 'all coordinates used at least once' and 'all coordinates used once and only once' happen to describe the same 3 points. They come apart as soon as the axes have different lengths. On a 4x7x10 grid, 'once and only once' has no solution at all: a set of points that uses each of the 10 coordinates of the third axis exactly once has 10 points, and those 10 points cannot use each of the 4 coordinates of the first axis exactly once. **At least once** is the condition I actually require, and it is the one that generalises.

So: how few data points can use every coordinate at least once? Each data point sits at one site, and so supplies exactly one coordinate on each axis. Covering an axis with L coordinates therefore needs at least L data points, and the longest axis sets the floor:

```
    min number of data points  =  max(shape)

    min density  =  max(shape) / prod(shape)
```

where `shape` is the list of axis lengths, `max(shape)` the longest axis, and `prod(shape)` the total number of sites.

That floor is reachable, not just a lower limit. Walk the longest axis in order, one data point per coordinate, and for each shorter axis repeat its coordinates as many times as needed to fill the same number of points. Every axis is then covered by `max(shape)` points. On the 4x7x10 grid this gives 10 data points, using the third axis once per coordinate, the second axis with 3 coordinates repeated, and the first with 2 coordinates repeated.

The cubic case is the special case where all axes are equal, and there the formula reduces to the familiar `1/n^(d-1)` for a grid of `d` dimensions each of size `n`, since `n^d / n^(d-1) = n`:

```
    grid          min data points   min density
    3x3                         3        1/3
    10x10x10                   10       1/100
    4x7x10                     10       10/280
    2x3x5x7                     7        7/210
    50x10x1x1                  50       50/500
```

Two things follow that are easy to get wrong. The bound is set by the **longest** axis, not the shortest — intuition tends to reach for the shortest. And an axis of length 1 costs nothing: it contributes one coordinate, which any single data point already covers, so `50x10x1x1` needs the same 50 points as `50x10`.

^2 pun intended

^3 this is the later assumption mentioned in ^1

### Tabular vs array

This is the only real, sharp, dicotomy of the three, and it identifies that data structure (in memory or on disk), which is either tabular or array-like.

Tabular means that each data point is a row of a table, which has as many columns as the number of dimensions of the data points plus the number variables hosted throughout the data points. For example: assuming that the points in the 3x3 grid of [TD ADD FIGURE] are measurements of air temperature and humidity, to represent them in a table we need 4 columns: 2 for the coordinates and 2 for the variables. If not all the points have measurements of both temperature and humidty, little matters and I still need 4 columns as long as at least one point measure one variable and another point another. How many rows do I need? As many as the points (i.e. occupied sites), so 9 in our examples. The total number of values stored is n_cols x n_rows (= 36 in our example).

Array-like data structures are a bit more complex (but still quite intuitive). They generally store as many 1D arrays as there dimensions, and as many nD arrays as there are variables (where n potentially varies per variable and depends on the dimensions on which each variable is measured). The dimensions arrays contains the coordinates that define the grid. In the 3x3 grid of [TD ADD FIGURE] we'd have 2 dimensions arrays (lat = [lat1, lat2, lat3] and lon = [lon1, lon2, lon3]) and 2 variables arrays (temperature = [temp11, temp12, temp13; temp21, temp22, temp23; temp31, temp32, temp33], similar for humidity). The total number of values stored in this case is 24 (6 for the dimensions, 18 for the variables). Note that the variables arrays are linked to the dimensions arrays, such that it's fast to access the variable at a given location.

It's easy to see from the example that the array-like structure requires fewer data to fully describe the points of a fully occupied grid. On the other hand, the tabular structure requires fewer data for sparse cases. In the most sparse 3x3 case, a table requires 12 values: n_rows = 3 occupied points, and n_cols = 4 (assuming again two variables). An array instead still requires 24: because of the link between variables arrays and their dimensions, missing variables values at vacant sites must be recorded (e.g. as NaNs).

In brief: 

- the strength of array-like structures is to store all the coordinates values only once, but their weakness is to require and store a variable value at every site (occupied or vacant) described by the coordinates;
- the weakness of tabular structures is to store all coordinates values as many times as they are occupied, but their strength is to store records only of occupied sites.

#### parquet/pandas vs netCDF/xarray (and zarr)

To the above tabular and array structures correspond in-memory and on-disk representations. The most common on-disk formats are probably csv and parquet for tabular data, and netCDF and zarr for array structures. In the Python world, tabular data are generally handled with the `pandas` library, while array data are handled with `xarray`.  

### Mixed terminology

#### Site

Here, a site is a set of coordinates on the grid, i.e. a grid point. I try not to use the word point to prevent confusion with data point.

#### Variable

Any physical measurement or any other record at a site: temperature, date, names, birthday, favourite colour, etc depending on what type of data you're working on.

#### Observation

It's the record of a variable at a site + the site's coordinates.

#### Data point

It's an occupied site, i.e. a grid point to which one or more variable values (e.g. temperature, date, etc.) is associated. A data point contains as many observations as variables have values there.
