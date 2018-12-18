# MIXSAT

*A MAXSAT solver by low-rank SDP and branch-and-bound. Crafted by [Po-Wei Wang](http://powei.tw) and
[J. Zico Kolter](http://zicokolter.com).*

---

+ [More details from our paper (AAAI'19)](https://arxiv.org/abs/1812.06362)


### Compilation
The directory contains the source code for the MIXSAT solvers:

complete.c: The MIXSAT solver for complete track, will output solutions only after verification.

incomplete.c: The MIXSAT solver for incomplete track, output solutions once immediately without verification.

To compile the code, please type
>	 make

The solvers can solve any unweighted DIMACS CNF file by
>	 ./complete FILE  
>	 ./incomplete FILE
