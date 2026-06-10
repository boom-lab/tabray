# Tutorials and Explainers

## Data points: sparse vs dense, irregular vs gridded, tabular vs array

A set of data points generally lies on a spectrum between irregular and regular. While I have not defined yet what regular mean in the present context, when you read the previous sentence your mind probably used one of three images to picture the two extremes: sparse vs dense (a cloud with many points vs an almost empty region), irregular vs gridded (few spread out points vs several regularly spaced points), or tabular vs array (a list of records vs points identified by their coordinates).

While it's easy to see similarities among sparse/irregular/tabular and dense/gridded/array, some differences exist and so I'll define them in the context of this work for the sake of a deeper understanding of what I set out to do with it.

### Irregular vs gridded

If a set of points is spaced such that we can identify a set of dimensions whose coordinates are regularly spaced, then I call the dataset **gridded**. Example: a set of 9 points on a 3x3 grid with latitude and longitude coordinates, and the points are 1 degree latitude and longitude apart. [TD ADD FIGURE] In this case, I call the data *purely gridded* because the data occupies all available sites (i.e. longitude-latitude pairs).

The opposite end of purely gridded is *purely irregular*: in this case we cannot identify a set of dimensions whose coordinates are regularly spaced where the points are located. For example, 3 points on a 2D space and whose coordinates are all different^1 [TD ADD FIGURE]

Anything in between has some degree of both, so I refer to it as gridded or irregular depending on what extreme they look closest to to me, purely based on my gut feeling. This is heavily based on the sparsity of the data, so let's talk about it next.

^1 to be rigorous the coordinates of the occupied sites would also have to be incommensurable or one can generate a grid with empty sites that satisfies the purely gridded definition, but because of some later assumptions on data sparsity I stick to this less rigorous definition that makes communication easier.

### Sparse vs dense

I define sparsity as the ratio of sites not occupied by data points. By contrast, density (or occupancy) is the ratio of sites occupied by data points. Note that sparsity+density = 1.

Unambiguously, when all sites are occupied (e.g. TD ADD FIGURE) density = 1 and sparsity = 0.

Sparsity = 1 (density = 0) also unambiguosly identifies the case where all sites are vacant, but it's pretty useless in practice (I want to investigate different storage techniques, and I need data to store).

The most interesting largest sparsity (smallest density) value is the one such that all available coordinates are used once and only once. For example: 3 points on a 3x3 grid, such that no two points share the same coordinates [TD ADD FIGURE]. P1 is located at (lon1,lat1); P2 at (lon3,lat2); P3 at (lon2,lat3). In this case sparsity = 6/9 = 2/3 (density = 1/3). Note that this is the largest sparsity possible for this 3x3 grid: adding any point would lower it (e.g. TD ADD PANEL TO FIGURE), and removing any point would make 2 coordinates pointless^2 (e.g. TD ADD PANEL TO FIGURE), because their knowledge does not add anything, and I don't want to store data that is not required to describe my data points.

This is why I only consider cases where all coordinates are occupied at least once^3. I try to refer to cases like [TD ADD FIGURE] as 'least dense' or 'most sparse', but I might accidentally call them 'purely irregular' or 'irregular', as they are the most irregular I consider in this work.

Note that the value of largest sparsity depends on the grid (number of dimensions and their sizes).

^2 pun intended

^3 this is the later assumption mentioned in ^1

