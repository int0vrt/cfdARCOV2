/*
cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
Copyright (C) 2024 cfdARCHO

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef CFDARCO_EIGEN_HANDLER_HPP
#define CFDARCO_EIGEN_HANDLER_HPP

#include <Eigen/Dense>
#include <memory>

template<class ArgType, typename... Ptrs>
class Holder;

namespace Eigen {
    namespace internal {
        template<class ArgType, typename... Ptrs>
        struct traits<Holder < ArgType, Ptrs...>> {
        typedef typename ArgType::StorageKind StorageKind;
        typedef typename traits<ArgType>::XprKind XprKind;
        typedef typename ArgType::StorageIndex StorageIndex;
        typedef typename ArgType::Scalar Scalar;
        enum {
            Flags = ArgType::Flags & RowMajorBit,
            RowsAtCompileTime = ArgType::RowsAtCompileTime,
            ColsAtCompileTime = ArgType::ColsAtCompileTime,
            MaxRowsAtCompileTime = ArgType::MaxRowsAtCompileTime,
            MaxColsAtCompileTime = ArgType::MaxColsAtCompileTime
        };
    };
}  // namespace internal
}  // namespace Eigen

template<typename ArgType, typename... Ptrs>
class Holder
        : public Eigen::internal::dense_xpr_base<Holder<ArgType, Ptrs...>>::type {
public:
    Holder(const ArgType &arg, Ptrs *... pointers)
            : m_arg(arg), m_unique_ptrs(std::unique_ptr<Ptrs>(pointers)...) {}

    typedef typename Eigen::internal::ref_selector<Holder<ArgType, Ptrs...>>::type
            Nested;
    typedef Eigen::Index Index;

    Index rows() const { return m_arg.rows(); }

    Index cols() const { return m_arg.cols(); }

    typedef typename Eigen::internal::ref_selector<ArgType>::type ArgTypeNested;
    ArgTypeNested m_arg;
    std::tuple<std::unique_ptr<Ptrs>...> m_unique_ptrs;
};

namespace Eigen {
    namespace internal {
        template<typename ArgType, typename... Ptrs>
        struct evaluator<Holder < ArgType, Ptrs...>> : evaluator_base<Holder < ArgType>> {
        typedef Holder<ArgType, Ptrs...> XprType;
        typedef typename nested_eval<ArgType, 1>::type ArgTypeNested;
        typedef typename remove_all<ArgTypeNested>::type ArgTypeNestedCleaned;
        typedef typename XprType::CoeffReturnType CoeffReturnType;
        enum {
            CoeffReadCost = evaluator<ArgTypeNestedCleaned>::CoeffReadCost,
            Flags = evaluator<ArgTypeNestedCleaned>::Flags
                    & (HereditaryBits | LinearAccessBit | PacketAccessBit),
            Alignment = Unaligned & evaluator<ArgTypeNestedCleaned>::Alignment,
        };

        evaluator<ArgTypeNestedCleaned> m_argImpl;

        evaluator(const XprType &xpr) : m_argImpl(xpr.m_arg) {}

        EIGEN_STRONG_INLINE CoeffReturnType coeff(Index row, Index col) const {
            CoeffReturnType val = m_argImpl.coeff(row, col);
            return val;
        }

        EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
            CoeffReturnType val = m_argImpl.coeff(index);
            return val;
        }

        template<int LoadMode, typename PacketType>
        EIGEN_STRONG_INLINE PacketType packet(Index row, Index col) const {
            return m_argImpl.template packet<LoadMode, PacketType>(row, col);
        }

        template<int LoadMode, typename PacketType>
        EIGEN_STRONG_INLINE PacketType packet(Index index) const {
            return m_argImpl.template packet<LoadMode, PacketType>(index);
        }
    };
}  // namespace internal
}  // namespace Eigen

template<typename T, typename... Ptrs>
Holder<T, Ptrs...> makeHolder(const T &arg, Ptrs *... pointers) {
    return Holder<T, Ptrs...>(arg, pointers...);
}

#endif //CFDARCO_EIGEN_HANDLER_HPP
