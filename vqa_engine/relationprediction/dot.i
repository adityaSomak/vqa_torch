%module dot

%{
    #define SWIG_FILE_WITH_INIT
    #include "dot.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {(int len1, double* vec1), (int len2, double* vec2)}


%include "dot.h"
%rename (dot) my_dot;

%inline %{
    double my_dot(int len1, double* vec1, int len2, double* vec2) {
    if (len1 != len2) {
        PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given", len1, len2);
        return 0.0;
    }
    return dot(len1, vec1, vec2);
}
%}