package main

type box struct {
	Low      []float64
	Up       []float64
	Elems    []Element
	children []*box
}

func (b *box) contains(e Element) bool {
	low, up := e.Bounds()
	for i := range low {
		center := low[i] + (up[i]-low[i])/2
		if center < b.Low[i] || b.Up[i] < center {
			return false
		}
	}
	return true
}

func (b *box) Find(x []float64) Element {
	for _, child := range b.children {
		for i := range x {
			if child.Low[i] <= x[i] && x[i] <= child.Up[i] {
				return child.Find(x)
			}
		}
		return childchild
	}
}

func (b *box) split(elemTarget, nsplit int) {
	if len(b.Elems) < 100 {
		return
	}

	b.splitBox(nsplit)
	for _, elem := range b.Elems {
		for _, child := range b.children {
			if child.contains(elem) {
				child.Elems = append(child.Elems, elem)
				break
			}
		}
	}

	for _, child := range b.children {
		child.split(elemTarget, nsplit)
	}
}

func newBox(elems []Element) *box {
	lowest, upest := elems[0].Bounds()
	for _, elem := range elems[1:] {
		low, up := elem.Bounds()
		for i := range low {
			if low[i] < lowest[i] {
				lowest[i] = low[i]
			}
			if up[i] > upest[i] {
				upest[i] = up[i]
			}
		}
	}

	nsplit := 3
	b := &box{Low: lowest, Up: upest, Elems: elems}
	b.split(100, nsplit)
	return b
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

func (b *box) splitBox(n int) {
	ndim := len(b.Low)
	combs := combinations(ndim, n, nil)
	b.children = make([]*box, len(combs))
	for i, comb := range combs {
		b.children[i] = &box{}
		sub := b.children[i]
		sub.Low = make([]float64, ndim)
		sub.Up = make([]float64, ndim)
		for dim, section := range comb {
			dx := b.Up[dim] - b.Low[dim]
			sub.Low[dim] = b.Low[dim] + dx/float64(n)*float64(section)
			sub.Up[dim] = sub.Low[dim] + dx/float64(n)
		}
	}
}
