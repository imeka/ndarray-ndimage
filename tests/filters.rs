use ndarray_image::{median_filter, Mask};

#[test] // Results verified manually.
fn test_median_filter() {
    let mut gt = Mask::from_elem((3, 3, 3), false);
    let mut mask = gt.clone();
    mask[(0, 0, 0)] = true;
    assert_eq!(median_filter(&mask), gt);
    mask[(1, 0, 0)] = true;
    assert_eq!(median_filter(&mask), gt);
    mask[(0, 1, 0)] = true;
    assert_eq!(median_filter(&mask), gt);

    gt[(0, 0, 0)] = true;
    mask[(0, 0, 1)] = true;
    assert_eq!(median_filter(&mask), gt);

    mask[(1, 1, 0)] = true;
    assert_eq!(median_filter(&mask), gt);

    gt[(1, 0, 0)] = true;
    gt[(0, 1, 0)] = true;
    gt[(0, 0, 1)] = true;
    mask[(1, 0, 1)] = true;
    assert_eq!(median_filter(&mask), gt);

    gt[(2, 0, 0)] = true;
    mask[(1, 1, 1)] = true;
    assert_eq!(median_filter(&mask), gt);
}
