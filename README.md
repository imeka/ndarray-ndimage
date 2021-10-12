ndarray-image
=============

This crate provides multidimensional image processing for [`ndarray`]'s `ArrayBase` type.

It aims to:
- be a Rust replacement for [`scipy.ndimage`] with some other tools like [`numpy.pad`] and anything else relevant to image processing.
- be faster or as fast as `scipy.ndimage`. Most of it is cythonized so it's not easy as it seems.
- avoid using `unsafe`. This is not a unbreakable rule. Its usage will be evaluated and dicussed in the pull requests.

Currently available routines include:
- Filters: median_filter
- Fourier filters: none
- Interpolation: spline_filter, spline_filter1d
- Measurements: label
- Morphology: binary_dilation, binary_erosion
- Padding: reflect, symmetric and wrap

**This crate is a work-in-progress.** Some functions are provided but they offer less options than SciPy. Some are offered only in 3D, with less boundary modes, with only 2 types of structuring element, only for binary data, only for f64, etc.

[`ndarray`]: https://github.com/rust-ndarray/ndarray
[`scipy.ndimage`]: https://docs.scipy.org/doc/scipy/reference/ndimage.html
[`numpy.pad`]: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

Using with Cargo
================

```toml
[dependencies]
ndarray = "0.15"
ndarray-image = "0.1"
```

Contributing
============

`ndarray-image` needs your help to grow. Please feel free to create issues and submit PRs. Since it is based on `scipy.ndimage`, it is easy to port new functions and tests.

License
=======

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.