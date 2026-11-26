# Linear Algebra — Lecture Notes, Weeks 7–10

## Week 7: Eigenvalues and eigenvectors

A non-zero vector v is an eigenvector of a square matrix A if Av = λv for some scalar λ, the eigenvalue. Eigenvalues are the roots of the characteristic polynomial det(A − λI) = 0. An n × n matrix has n eigenvalues counted with multiplicity, possibly complex.

The algebraic multiplicity of λ is its multiplicity as a root; the geometric multiplicity is the dimension of the eigenspace, the null space of A − λI. The geometric multiplicity is at least 1 and at most the algebraic multiplicity.

Useful facts: the trace of A equals the sum of its eigenvalues, and the determinant equals their product. A triangular matrix has its diagonal entries as eigenvalues.

## Week 8: Diagonalisation

A is diagonalisable if A = PDP⁻¹ with D diagonal; the columns of P are eigenvectors and the diagonal of D holds the eigenvalues. A is diagonalisable exactly when, for every eigenvalue, the geometric multiplicity equals the algebraic multiplicity — equivalently, when A has n linearly independent eigenvectors. A matrix with n distinct eigenvalues is always diagonalisable.

Diagonalisation makes powers easy: Aᵏ = PDᵏP⁻¹. This is how we solved the Fibonacci recurrence in the tutorial, with eigenvalues (1 ± √5)/2.

The matrix [[1, 1], [0, 1]] is the standard example of a matrix that is not diagonalisable: its only eigenvalue is 1 with algebraic multiplicity 2 but geometric multiplicity 1.

## Week 8 (continued): Symmetric matrices

The spectral theorem: every real symmetric matrix has real eigenvalues and can be diagonalised by an orthogonal matrix, A = QΛQᵀ. Eigenvectors for different eigenvalues are orthogonal. A symmetric matrix is positive definite exactly when all its eigenvalues are positive.

## Week 9: Singular value decomposition

Every real m × n matrix A, square or not, has a singular value decomposition A = UΣVᵀ, where U (m × m) and V (n × n) are orthogonal and Σ is m × n with non-negative entries σ₁ ≥ σ₂ ≥ … ≥ 0 on its diagonal. The singular values are the square roots of the eigenvalues of AᵀA. The columns of V are eigenvectors of AᵀA, and the columns of U are eigenvectors of AAᵀ.

The rank of A equals the number of non-zero singular values. Truncating the SVD to the k largest singular values gives the best rank-k approximation in both the spectral and Frobenius norms (the Eckart–Young theorem). This is the basis of image compression and of principal component analysis.

## Week 10: Least squares

When Ax = b has no solution, the least-squares solution minimises ‖Ax − b‖. It satisfies the normal equations AᵀAx = Aᵀb. If A has independent columns, AᵀA is invertible and x = (AᵀA)⁻¹Aᵀb. Numerically it is better to use a QR decomposition or the SVD than to form AᵀA, which squares the condition number.

Fitting a straight line y = c + mx to data points is the standard example: each point gives one row [1, xᵢ] of A.

## Week 10 (continued): Principal component analysis

Centre the data matrix X (subtract each column's mean). The principal directions are the right singular vectors of X, and the variance explained by component i is σᵢ² / (n − 1). Keeping the first k components gives the k-dimensional projection that preserves the most variance.

## Exam notes

The exam is on 9 December, 09:30, in the main hall, 2 hours. One A4 sheet of handwritten notes, both sides, is allowed. No calculators. Questions on weeks 1–10; weeks 7–10 carry 60 % of the marks.
