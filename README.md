ndarray-ndimage
=============

This crate provides multidimensional image processing for [`ndarray`]'s `ArrayBase` type. It is mainly focussed on 3D arrays/images for now, but some functions are available on on n-d arrays.

It aims to:
- be a Rust replacement for [`scipy.ndimage`] with some other tools like [`numpy.pad`] and anything else relevant to image processing. We do not want all options and arguments offered by `scipy.ndimage` because some of them are incompatible with Rust. We hope to offer the most used ones.
- be faster or as fast as `scipy.ndimage`. Most of it is cythonized so it's not as easy as it seems. In fact, I'm usually unable to be faster than SciPy but it does happen on some functions.
- avoid using `unsafe`. This is not an unbreakable rule. Its usage will be evaluated and dicussed in the pull requests.

Currently available routines include:
- Filters: convolve/1d, correlate/1d, gaussian_filter/1d, min/max_filter/1d, median_filter, prewitt, sobel
- Fourier filters: none. Please use the excellent [`rustfft`] crate
- Interpolation: spline_filter/1d
- Measurements: label, label_histogram, largest_connected_components, most_frequent_label
- Morphology: binary_closing, binary_dilation, binary_erosion, binary_opening. Works on all kernels (structuring elements).
- Padding: Almost all modes. Work for all dimensions and types.

**This crate is a work-in-progress.** Only a subset of the `scipy.ndimage` functions are provided and most of them offer less options than SciPy. Some are offered only in 3D, with less boundary modes, with only 2 types of structuring element, only for binary data, only for f64, etc.

[`ndarray`]: https://github.com/rust-ndarray/ndarray
[`scipy.ndimage`]: https://docs.scipy.org/doc/scipy/reference/ndimage.html
[`numpy.pad`]: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
[`rustfft`]: https://crates.io/crates/rustfft

Using with Cargo
================

```toml
[dependencies]
ndarray = "0.15"
ndarray-ndimage = "0.2"
```

Contributing
============

`ndarray-ndimage` needs your help to grow. Please feel free to create issues and submit PRs. Since it is based on `scipy.ndimage`, it is easy to port new functions and tests. Reading Cython code is highly unpleasant; the joy comes from porting it to Rust!

License
=======

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.
