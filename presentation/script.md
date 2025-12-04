<!-- 5 minutes only!!!! -->
<!-- start with intro/brief background from oli, 2.5 minutes -->
<!-- then alex will talk about brief implementation and results/conclusions, 2.5 minutes -->

# intro: oli (~2.5 min)

## what is qram? (~45 sec)

<!-- [slide: title slide - "fat-tree qram: a high-bandwidth shared quantum memory architecture"] -->

hi! today we're presenting our work reproducing and validating fat-tree qram.

so what is qram? quantum random access memory lets quantum algorithms query a classical database in superposition. you give it an address register in superposition, and it returns the corresponding data values entangled with each address.

this is critical because many important quantum algorithms—grover search, hhl for linear systems, quantum machine learning—all assume qram exists and just count queries. without efficient qram, these algorithms can't deliver their promised speedups.

---

## bucket-brigade qram (~1 min 15 sec)

the standard qram design is bucket-brigade qram. the key building block is a quantum router.

<!-- [slide: show bb_qram_figure.jpg part (b) - single router diagram] -->

each router has four ports—input at the top, a route register that stores a routing decision, and left/right outputs going to child routers. the key operation is the controlled-swap: if the route bit is 0, we swap the input toward the left child; if it's 1, toward the right. so a router is just a quantum switch that steers qubits left or right based on a stored bit.

<!-- [slide: show bb_qram_figure.jpg part (c) - the h-tree layout] -->

bucket-brigade arranges these routers in a binary tree above the memory cells. here's the layout for n=8—circles are routers, data cells are at the bottom. the red routers are "active"—meaning they've received an address bit and are ready to route. notice only one path from root to leaf is active, not the whole tree.

<!-- [slide: show bb_qram_figure.jpg part (a) - query execution phases] -->

this diagram shows how a query executes over time. reading left to right: first, address bits load down through the tree, each bit getting stored at successive levels to activate routers along one path. then a bus qubit descends to the leaves, picks up data via x gates, and returns. finally address bits unload. the whole thing takes o(log n) depth.

the key insight is that even for superposition queries, each branch only activates one path—so errors scale as o(log n), not o(n). this is what makes it practical.

---

## the bandwidth problem (~20 sec)

but there's a catch. when one query is running, it occupies routers along the entire root-to-leaf path. if you have multiple clients wanting to query simultaneously, they have to wait. for p queries, total time becomes o(p log n). qram becomes a bandwidth bottleneck.

---

## fat-tree solution (~25 sec)

<!-- [slide: show fat_tree_figure.jpg - the full fat-tree structure for n=32] -->

fat-tree qram solves this by adding extra copies of routers. you can see each node now contains multiple routers—the different colors show different "layers." a scheduling protocol lets queries move through these layers as they progress, so they don't block each other.

this means up to o(log n) queries can be in flight simultaneously, completing in o(log n) total time—giving you o(1) amortized latency per query. the trade-off is about n times more routers.

---

# implementation & results: alex (~2.5 min)

## implementation (~1 min)

we implemented both architectures in qiskit.

<!-- [slide: maybe show a code snippet or circuit diagram?] -->

the main challenges were:
- **simulation scalability**: fat-tree for n=4 needs over 150 qubits—way beyond statevector simulation. we validated on small instances.
- **translating the paper's algorithm**: the scheduling protocol was described abstractly. we had to figure out operation parallelism, when exactly to apply classical data, and how address bits get reused across pipelined queries.

<!-- alex: add more implementation details you want to mention here -->

---

## results / conclusion

<!-- alex: fill this in -->

---

