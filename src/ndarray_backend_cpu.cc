#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};

struct AlignedArrayIndex {
  AlignedArrayIndex(const std::vector<int32_t>& shape,
                    const std::vector<int32_t>& strides,
                    const size_t offset)
   : shape(shape), strides(strides), offset(offset), indexes(shape.size()), cur_axis(shape.size() - 1), index(offset) {
    assert(shape.size() == strides.size() && "size of shape is not equal to size of strides");
  }
  operator size_t() const {
    return index;
  }
  bool is_end() const {
    return cur_axis < 0;
  }
  void next() {
    index += strides[cur_axis];
    ++indexes[cur_axis];
    if (indexes[cur_axis] >= shape[cur_axis]) {
      while (indexes[cur_axis] >= shape[cur_axis]) {
        index -= strides[cur_axis] * shape[cur_axis];
        indexes[cur_axis] = 0;
        --cur_axis;
        if (is_end()) {
            return;
        }
        ++indexes[cur_axis];
        index += strides[cur_axis];
      }
      cur_axis = shape.size() - 1;
    }
  }

  private:
    std::vector<int32_t> shape;
    std::vector<int32_t> strides;
    size_t offset;

    std::vector<int32_t> indexes;
    int cur_axis;
    size_t index;
};

void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  size_t i = 0;
  AlignedArrayIndex index(shape, strides, offset);
  while(!index.is_end()) {
    out->ptr[i++] = a.ptr[index];
    index.next();
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t i = 0;
  AlignedArrayIndex index(shape, strides, offset);
  while(!index.is_end()) {
    out->ptr[index] = a.ptr[i++];
    index.next();
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  AlignedArrayIndex index(shape, strides, offset);
  for (size_t i = 0; i < out->size && !index.is_end(); ++i, index.next()) {
    out->ptr[index] = val;
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

template <typename Op>
void BiScalarOp(const AlignedArray& a, scalar_t val, AlignedArray* out, Op op) {
  assert(a.size == out->size && "size of a and out is not equal");
  for (size_t i = 0; i < a.size; ++i) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}
template <typename Op>
void BiEWiseOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, Op op) {
  assert(a.size == b.size && "size of a and b is not equal");
  assert(a.size == out->size && "size of a and out is not equal");
  size_t size = out->size;
  for (size_t i = 0; i < size; ++i) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}
template <typename Op>
void UniEWiseOp(const AlignedArray& a, AlignedArray* out, Op op) {
  assert(a.size == out->size && "size of a and out is not equal");
  size_t size = out->size;
  for (size_t i = 0; i < size; ++i) {
    out->ptr[i] = op(a.ptr[i]);
  }
}
void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  BiScalarOp(a, val, out, std::multiplies<scalar_t>());
}
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BiEWiseOp(a, b, out, std::multiplies<scalar_t>());
}
void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  BiScalarOp(a, val, out, std::divides<scalar_t>());
}
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BiEWiseOp(a, b, out, std::divides<scalar_t>());
}
void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  BiScalarOp(a, val, out, [](scalar_t x, scalar_t n) { return std::pow(x, n);});
}
void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  BiScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return std::max(x, y);});
}
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BiEWiseOp(a, b, out, [](scalar_t x, scalar_t y) { return std::max(x, y);});
}
void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  BiScalarOp(a, val, out, std::equal_to<scalar_t>());
}
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BiEWiseOp(a, b, out, std::equal_to<scalar_t>());
}
void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  BiScalarOp(a, val, out, std::greater_equal<scalar_t>());
}
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BiEWiseOp(a, b, out, std::greater_equal<scalar_t>());
}
void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  UniEWiseOp(a, out, [](scalar_t x) {return std::log(x);});
}
void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  UniEWiseOp(a, out, [](scalar_t x) {return std::exp(x);});
}
void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  UniEWiseOp(a, out, [](scalar_t x) {return std::tanh(x);});
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      size_t cur = i * p + j;
      out->ptr[cur] = 0;
      for (size_t k = 0; k < n; ++k) {
        out->ptr[cur] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (size_t i = 0; i < TILE; ++i) {
      for (size_t j = 0; j < TILE; ++j) {
        size_t cur = i * TILE + j;
        for (size_t k = 0; k < TILE; ++k) {
          out[cur] += a[i * TILE + k] * b[k * TILE + j];
        }
      }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; ++i) {
    out->ptr[i] = 0;
  }
  // calc
  for (size_t i = 0; i < m / TILE; ++i) {
    for (size_t j = 0; j < p / TILE; ++j) {
      for (size_t k = 0; k < n / TILE; ++k) {
        AlignedDot(
          a.ptr + i * n * TILE + k * TILE * TILE,
          b.ptr + k * p * TILE + j * TILE * TILE,
          out->ptr + i * p * TILE + j * TILE * TILE
        );
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  assert(a.size == out->size * reduce_size && "size of a with recude_size and out is not equal");
  size_t total_size = a.size;
  size_t after_reduce_size = out->size;
  for (size_t i = 0; i < after_reduce_size; ++i) {
    out->ptr[i] = a.ptr[i * reduce_size];
    for (size_t j = 0; j < reduce_size; ++j) {
      out->ptr[i] = std::max(out->ptr[i], a.ptr[i * reduce_size + j]);
    }
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  assert(a.size == out->size * reduce_size && "size of a with recude_size and out is not equal");
  size_t total_size = a.size;
  size_t after_reduce_size = out->size;
  for (size_t i = 0; i < after_reduce_size; ++i) {
    out->ptr[i] = 0;
    for (size_t j = 0; j < reduce_size; ++j) {
      out->ptr[i] += a.ptr[i * reduce_size + j];
    }
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
