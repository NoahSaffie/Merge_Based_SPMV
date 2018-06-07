.global singleMM		//global singleMM
	
singleMM:
	pushq %rbp
	movq %rsp, %rbp
	vmovsd %xmm0, -8(%rbp)		// Get first parameter (Csr Value)
	vmovsd %xmm1, -16(%rbp)		// Get second parameter (x value)
	vmulsd %xmm0, %xmm0, %xmm1	// Multiplication
	movq %rax, %xmm0		// Move result to return register
	popq %rbp
	ret
	
