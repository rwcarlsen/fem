package main

type box struct {
	Low []float64
	Up  []float64
}

func newBox(elems []Element) *box {
	//for dim := range
	panic("unimplemented")
}

func combinations(ndims, nsplits int, prefix []int) [][]int {
	if len(prefix) == ndims {
		return [][]int{prefix}
	}

	combs := [][]int{}
	for i := 0; i < nsplits; i++ {
		combs = append(combs, combinations(ndims, nsplits, append(prefix, i))...)
	}
	return combs
}

func splitBox(b *box, n int) []*box {
	ndim := len(b.Low)
	combs := combinations(ndim, n, nil)
	boxes := make([]*box, len(combs))
	for i, comb := range combs {
		boxes[i] = &box{}
		sub := boxes[i]
		sub.Low = make([]float64, ndim)
		sub.Up = make([]float64, ndim)
		for dim, section := range comb {
			dx := b.Up[dim] - b.Low[dim]
			sub.Low[dim] = b.Low[dim] + dx/float64(n)*float64(section)
			sub.Up[dim] = sub.Low[dim] + dx/float64(n)
		}
	}
	return boxes
}
