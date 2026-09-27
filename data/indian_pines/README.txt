Indian Pines hyperspectral benchmark
====================================

Origin: 220-band AVIRIS image of the Indian Pine test site 3, acquired 12 June
1992; the 200-band "corrected" version and its 16-class ground-truth map are
distributed by Purdue University (Baumgardner, Biehl and Landgrebe,
doi 10.4231/R7RX991C).

Files
-----
Indian_pines_corrected.mat : variable "indian_pines_corrected", 145 x 145 x 200, uint16
Indian_pines_gt.mat        : variable "indian_pines_gt", 145 x 145, uint8,
                             class labels 0 (unlabeled) through 16

Only the 10,249 pixels with a nonzero ground-truth label are used in the
manuscript (Section 3.3, Table for the labeled hyperspectral benchmark).

Provenance note
---------------
These two files were downloaded from the public MADANet mirror
(https://raw.githubusercontent.com/wjxcs/MADANet/master/), which serves the
same arrays as the Purdue distribution. The reproduction code reads the
bundled copies and does not require a network connection.

License and terms of use of the original dataset are those of the Purdue
Research Repository; the arrays are redistributed here unmodified.
