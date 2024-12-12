#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

// Definitions for function evaluation
#define MAX_EXPR_SIZE 256
#define MAX_STACK_SIZE 256

typedef enum { NUMBER, VARIABLE, OPERATOR, FUNCTION, LEFT_PAREN, RIGHT_PAREN, COMMA } TokenType;

typedef struct {
    TokenType type;
    double value;
    char op;
    char func[10];
} Token;

typedef struct {
    Token* tokens;
    int size;
    int capacity;
} TokenArray;

typedef struct {
    Token* stack;
    int size;
    int capacity;
} TokenStack;

typedef struct {
    double* stack;
    int size;
    int capacity;
} DoubleStack;

TokenArray* create_token_array(int capacity) {
    TokenArray* arr = (TokenArray*)malloc(sizeof(TokenArray));
    arr->tokens = (Token*)malloc(capacity * sizeof(Token));
    arr->size = 0;
    arr->capacity = capacity;
    return arr;
}

void free_token_array(TokenArray* arr) {
    free(arr->tokens);
    free(arr);
}

TokenStack* create_token_stack(int capacity) {
    TokenStack* stack = (TokenStack*)malloc(sizeof(TokenStack));
    stack->stack = (Token*)malloc(capacity * sizeof(Token));
    stack->size = 0;
    stack->capacity = capacity;
    return stack;
}

void free_token_stack(TokenStack* stack) {
    free(stack->stack);
    free(stack);
}

DoubleStack* create_double_stack(int capacity) {
    DoubleStack* stack = (DoubleStack*)malloc(sizeof(DoubleStack));
    stack->stack = (double*)malloc(capacity * sizeof(double));
    stack->size = 0;
    stack->capacity = capacity;
    return stack;
}

void free_double_stack(DoubleStack* stack) {
    free(stack->stack);
    free(stack);
}

void push_token(TokenStack* stack, Token token) {
    if (stack->size < stack->capacity) {
        stack->stack[stack->size++] = token;
    }
}

Token pop_token(TokenStack* stack) {
    if (stack->size > 0) {
        return stack->stack[--stack->size];
    }
    Token empty_token = {0};
    return empty_token;
}

Token peek_token(TokenStack* stack) {
    if (stack->size > 0) {
        return stack->stack[stack->size - 1];
    }
    Token empty_token = {0};
    return empty_token;
}

void push_double(DoubleStack* stack, double value) {
    if (stack->size < stack->capacity) {
        stack->stack[stack->size++] = value;
    }
}

double pop_double(DoubleStack* stack) {
    if (stack->size > 0) {
        return stack->stack[--stack->size];
    }
    return 0.0;
}

int precedence(char op) {
    switch (op) {
        case '+':
        case '-': return 1;
        case '*':
        case '/': return 2;
        case '^': return 3;
        default: return 0;
    }
}

int is_operator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
}

int is_function(const char* str) {
    return strcmp(str, "sin") == 0 || strcmp(str, "cos") == 0 || strcmp(str, "tan") == 0 || strcmp(str, "cot") == 0 ||
           strcmp(str, "asin") == 0 || strcmp(str, "acos") == 0 || strcmp(str, "atan") == 0 || strcmp(str, "acot") == 0 || 
           strcmp(str, "log_") == 0;
}

void tokenize(const char* expr, TokenArray* tokens) {
    int i = 0;
    while (*expr) {
        if (isspace(*expr)) {
            expr++;
            continue;
        }

        if (isdigit(*expr) || (*expr == '.' && isdigit(*(expr + 1)))) {
            char* end;
            tokens->tokens[i].type = NUMBER;
            tokens->tokens[i].value = strtod(expr, &end);
            expr = end;
        } else if (*expr == 'e') {
            tokens->tokens[i].type = NUMBER;
            tokens->tokens[i].value = 2.718281828459;
            expr++;
        } else if (*expr == 'x') {
            tokens->tokens[i].type = VARIABLE;
            tokens->tokens[i].op = *expr;
            expr++;
        } else if (is_operator(*expr)) {
            tokens->tokens[i].type = OPERATOR;
            tokens->tokens[i].op = *expr;
            expr++;
        } else if (*expr == '(') {
            tokens->tokens[i].type = LEFT_PAREN;
            tokens->tokens[i].op = *expr;
            expr++;
        } else if (*expr == ')') {
            tokens->tokens[i].type = RIGHT_PAREN;
            tokens->tokens[i].op = *expr;
            expr++;
        } else if (*expr == ',') {
            tokens->tokens[i].type = COMMA;
            tokens->tokens[i].op = *expr;
            expr++;
        } else {
            char func[10];
            int j = 0;
            while ((isalpha(*expr) || *expr == '_') && j < 9) {
                func[j++] = *expr++;
            }
            func[j] = '\0';
            if (is_function(func)) {
                tokens->tokens[i].type = FUNCTION;
                strcpy(tokens->tokens[i].func, func);
            }
        }
        i++;
        if (i >= tokens->capacity) {
            tokens->capacity *= 2;
            tokens->tokens = (Token*)realloc(tokens->tokens, tokens->capacity * sizeof(Token));
        }
    }
    tokens->size = i;
}

void to_postfix(TokenArray* infix, TokenArray* postfix) {
    TokenStack* stack = create_token_stack(MAX_STACK_SIZE);
    int i, j = 0;
    for (i = 0; i < infix->size; i++) {
        Token token = infix->tokens[i];
        if (token.type == NUMBER || token.type == VARIABLE) {
            postfix->tokens[j++] = token;
        } else if (token.type == FUNCTION) {
            push_token(stack, token);
        } else if (token.type == OPERATOR) {
            while (stack->size > 0 && precedence(peek_token(stack).op) >= precedence(token.op)) {
                postfix->tokens[j++] = pop_token(stack);
            }
            push_token(stack, token);
        } else if (token.type == LEFT_PAREN) {
            push_token(stack, token);
        } else if (token.type == RIGHT_PAREN) {
            while (stack->size > 0 && peek_token(stack).type != LEFT_PAREN) {
                postfix->tokens[j++] = pop_token(stack);
            }
            pop_token(stack); // Pop the left parenthesis
            if (stack->size > 0 && peek_token(stack).type == FUNCTION) {
                postfix->tokens[j++] = pop_token(stack);
            }
        } else if (token.type == COMMA) {
            while (stack->size > 0 && peek_token(stack).type != LEFT_PAREN) {
                postfix->tokens[j++] = pop_token(stack);
            }
        }
    }
    while (stack->size > 0) {
        postfix->tokens[j++] = pop_token(stack);
    }
    postfix->size = j;
    free_token_stack(stack);
}

double evaluate_postfix(TokenArray* postfix, double x) {
    DoubleStack* stack = create_double_stack(MAX_STACK_SIZE);
    int i;
    for (i = 0; i < postfix->size; i++) {
        Token token = postfix->tokens[i];
        if (token.type == NUMBER) {
            push_double(stack, token.value);
        } else if (token.type == VARIABLE) {
            push_double(stack, x);
        } else if (token.type == OPERATOR) {
            double b = pop_double(stack);
            double a = pop_double(stack);
            switch (token.op) {
                case '+': push_double(stack, a + b); break;
                case '-': push_double(stack, a - b); break;
                case '*': push_double(stack, a * b); break;
                case '/': push_double(stack, a / b); break;
                case '^': push_double(stack, pow(a, b)); break;
            }
        } else if (token.type == FUNCTION) {
            double a = pop_double(stack);
            if (strcmp(token.func, "sin") == 0) {
                push_double(stack, sin(a));
            } else if (strcmp(token.func, "cos") == 0) {
                push_double(stack, cos(a));
            } else if (strcmp(token.func, "tan") == 0) {
                push_double(stack, tan(a));
            } else if (strcmp(token.func, "cot") == 0) {
                push_double(stack, 1/tan(a));
            } else if (strcmp(token.func, "asin") == 0) {
                push_double(stack, asin(a));
            } else if (strcmp(token.func, "acos") == 0) {
                push_double(stack, acos(a));
            } else if (strcmp(token.func, "atan") == 0) {
                push_double(stack, atan(a));
            } else if (strcmp(token.func, "acot") == 0) {
                push_double(stack, atan(1/a));
            } else if (strcmp(token.func, "log_") == 0) {
                double base = pop_double(stack);
                push_double(stack, log(a) / log(base));
            }
        }
    }
    double result = pop_double(stack);
    free_double_stack(stack);
    return result;
}

double evaluate_func(const char* expr, double x) {
    TokenArray* infix = create_token_array(strlen(expr));
    TokenArray* postfix = create_token_array(strlen(expr));
    tokenize(expr, infix);
    to_postfix(infix, postfix);
    double result = evaluate_postfix(postfix, x);
    free_token_array(infix);
    free_token_array(postfix);
    return result;
}

// Definitions for numerical methods
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

// Function declarations
void bisectionMethod();
void regulaFalsiMethod();
void newtonRaphsonMethod();
void inverseMatrix();
void gaussElimination();
void gaussSeidel();
void numericalDifferentiation();
void simpsonsRule();
void trapezoidalRule();
void gregoryNewtonInterpolation();
Matrix inputMatrix(int rows, int cols);
void freeMatrix(Matrix m);
unsigned long long factorial(int n);

// Global variables for function expressions
char function_expr[MAX_EXPR_SIZE];

// Get the function from user input
void getFunctionInput() {
    printf("Enter the function: ");
    scanf(" %[^\n]s", function_expr);
}

int main() {
    int choice;
    do {
        printf("Select a numerical method to execute:\n");
        printf("1. Bisection Method\n");
        printf("2. Regula-Falsi Method\n");
        printf("3. Newton-Raphson Method\n");
        printf("4. Inverse of NxN Matrix\n");
        printf("5. Gauss Elimination\n");
        printf("6. Gauss-Seidel\n");
        printf("7. Numerical Differentiation\n");
        printf("8. Simpson's Rule\n");
        printf("9. Trapezoidal Rule\n");
        printf("10. Gregory-Newton Interpolation\n");
        printf("0. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch(choice) {
            case 1: getFunctionInput(); bisectionMethod(); break;
            case 2: getFunctionInput(); regulaFalsiMethod(); break;
            case 3: getFunctionInput(); newtonRaphsonMethod(); break;
            case 4: inverseMatrix(); break;
            case 5: gaussElimination(); break;
            case 6: gaussSeidel(); break;
            case 7: getFunctionInput(); numericalDifferentiation(); break;
            case 8: getFunctionInput(); simpsonsRule(); break;
            case 9: getFunctionInput(); trapezoidalRule(); break;
            case 10: gregoryNewtonInterpolation(); break;
            case 0: printf("Exiting...\n"); break;
            default: printf("Invalid choice. Please try again.\n");
        }
    } while(choice != 0);
    return 0;
}

// Function implementations

Matrix inputMatrix(int rows, int cols) {
    Matrix m;
    int i;
    m.rows = rows;
    m.cols = cols;
    m.data = (double**)malloc(rows * sizeof(double*));
    if (m.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    for (i = 0; i < rows; i++) {
        m.data[i] = (double*)malloc(cols * sizeof(double));
        if (m.data[i] == NULL) {
            printf("Memory allocation failed.\n");
            exit(1);
        }
        for (int j = 0; j < cols; j++) {
            printf("Enter element [%d][%d]: ", i, j);
            scanf("%lf", &m.data[i][j]);
        }
    }
    return m;
}

void freeMatrix(Matrix m) {
    int i;
    for (i = 0; i < m.rows; i++) {
        free(m.data[i]);
    }
    free(m.data);
}

void bisectionMethod() {
    double a, b, tolerance;
    int maxIterations;
    
    printf("Enter the interval [a, b]: ");
    scanf("%lf %lf", &a, &b);
    printf("Enter the tolerance: ");
    scanf("%lf", &tolerance);
    printf("Enter the maximum number of iterations: ");
    scanf("%d", &maxIterations);

    int iter = 0;
    double c;

    printf("Iter\ta\t\tb\t\tc\t\tf(c)\n");
    while (iter < maxIterations) {
        c = (a + b) / 2;
        double fc = evaluate_func(function_expr, c);
        printf("%d\t%lf\t%lf\t%lf\t%lf\n", iter, a, b, c, fc);
        if (fc == 0 || (b - a) / 2 < tolerance) {
            printf("Root found: %lf\n", c);
            return;
        }
        iter++;
        if (fc * evaluate_func(function_expr, a) > 0) a = c;
        else b = c;
    }
    printf("Root not found within the given tolerance and maximum iterations.\n");
}

void regulaFalsiMethod() {
    double a, b, tolerance;
    int maxIterations;
    
    printf("Enter the interval [a, b]: ");
    scanf("%lf %lf", &a, &b);
    printf("Enter the tolerance: ");
    scanf("%lf", &tolerance);
    printf("Enter the maximum number of iterations: ");
    scanf("%d", &maxIterations);

    int iter = 0;
    double c;

    printf("Iter\ta\t\tb\t\tc\t\tf(c)\n");
    while (iter < maxIterations) {
        c = a - (evaluate_func(function_expr, a) * (b - a)) / (evaluate_func(function_expr, b) - evaluate_func(function_expr, a));
        double fc = evaluate_func(function_expr, c);
        printf("%d\t%lf\t%lf\t%lf\t%lf\n", iter, a, b, c, fc);
        if (fabs(fc) < tolerance) {
            printf("Root found: %lf\n", c);
            return;
        }
        iter++;
        if (fc * evaluate_func(function_expr, a) > 0) a = c;
        else b = c;
    }
    printf("Root not found within the given tolerance and maximum iterations.\n");
}

void newtonRaphsonMethod() {
    double x0, tolerance;
    int maxIterations;
    
    printf("Enter the initial guess: ");
    scanf("%lf", &x0);
    printf("Enter the tolerance: ");
    scanf("%lf", &tolerance);
    printf("Enter the maximum number of iterations: ");
    scanf("%d", &maxIterations);

    int iter = 0;
    double x1;
    double h = 1e-6;

    printf("Iter\tx0\t\tf(x0)\t\tx1\t\tf'(x0)\n");
    while (iter < maxIterations) {
        double fx0 = evaluate_func(function_expr, x0);
        double dfx0 = (evaluate_func(function_expr, x0 + h) - evaluate_func(function_expr, x0)) / h;
        x1 = x0 - fx0 / dfx0;
        printf("%d\t%lf\t%lf\t%lf\t%lf\n", iter, x0, fx0, x1, dfx0);
        if (fabs(x1 - x0) < tolerance) {
            printf("Root found: %lf\n", x1);
            return;
        }
        iter++;
        x0 = x1;
    }
    printf("Root not found within the given tolerance and maximum iterations.\n");
}

void inverseMatrix() {
    int i, j, k, n;
    printf("Enter the dimension of the matrix (n): ");
    scanf("%d", &n);

    Matrix m = inputMatrix(n, n);

    Matrix augmented;
    augmented.rows = n;
    augmented.cols = 2 * n;
    augmented.data = (double**)malloc(n * sizeof(double*));
    if (augmented.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    for (i = 0; i < n; i++) {
        augmented.data[i] = (double*)malloc(2 * n * sizeof(double));
        if (augmented.data[i] == NULL) {
            printf("Memory allocation failed.\n");
            exit(1);
        }
        for (j = 0; j < n; j++) {
            augmented.data[i][j] = m.data[i][j];
            augmented.data[i][j + n] = (i == j) ? 1 : 0;
        }
    }

    for (i = 0; i < n; i++) {
        double diagElement = augmented.data[i][i];
        if (diagElement == 0) {
            printf("Matrix is singular and cannot be inverted.\n");
            freeMatrix(augmented);
            freeMatrix(m);
            return;
        }
        for (j = 0; j < 2 * n; j++) {
            augmented.data[i][j] /= diagElement;
        }

        for (k = 0; k < n; k++) {
            if (k != i) {
                double factor = augmented.data[k][i];
                for (j = 0; j < 2 * n; j++) {
                    augmented.data[k][j] -= factor * augmented.data[i][j];
                }
            }
        }
    }

    Matrix inverse;
    inverse.rows = n;
    inverse.cols = n;
    inverse.data = (double**)malloc(n * sizeof(double*));
    if (inverse.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    for (i = 0; i < n; i++) {
        inverse.data[i] = (double*)malloc(n * sizeof(double));
        if (inverse.data[i] == NULL) {
            printf("Memory allocation failed.\n");
            exit(1);
        }
        for (j = 0; j < n; j++) {
            inverse.data[i][j] = augmented.data[i][j + n];
        }
    }

    printf("The inverse matrix is:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%lf ", inverse.data[i][j]);
        }
        printf("\n");
    }

    freeMatrix(augmented);
    freeMatrix(inverse);
    freeMatrix(m);
}

void gaussElimination() {
    int i, j, k, n, p, q;
    printf("Enter the number of equations: ");
    scanf("%d", &n);

    Matrix augmented = inputMatrix(n, n + 1);

    printf("Performing Gaussian elimination:\n");
    for (i = 0; i < n; i++) {
        for (k = i + 1; k < n; k++) {
            if (fabs(augmented.data[i][i]) < fabs(augmented.data[k][i])) {
                double* temp = augmented.data[i];
                augmented.data[i] = augmented.data[k];
                augmented.data[k] = temp;
            }
        }

        for (k = i + 1; k < n; k++) {
            double factor = augmented.data[k][i] / augmented.data[i][i];
            for (j = i; j <= n; j++) {
                augmented.data[k][j] -= factor * augmented.data[i][j];
            }
        }

        printf("After iteration %d:\n", i + 1);
        for (p = 0; p < n; p++) {
            for (q = 0; q <= n; q++) {
                printf("%lf ", augmented.data[p][q]);
            }
            printf("\n");
        }
    }

    double* solution = (double*)malloc(n * sizeof(double));
    if (solution == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    for (i = n - 1; i >= 0; i--) {
        solution[i] = augmented.data[i][n];
        for (j = i + 1; j < n; j++) {
            solution[i] -= augmented.data[i][j] * solution[j];
        }
        solution[i] /= augmented.data[i][i];
    }

    printf("The solution is:\n");
    for (i = 0; i < n; i++) {
        printf("x%d = %lf\n", i + 1, solution[i]);
    }

    free(solution);
    freeMatrix(augmented);
}

void gaussSeidel() {
    int i, j, n;
    printf("Enter the number of equations: ");
    scanf("%d", &n);

    Matrix augmented = inputMatrix(n, n + 1);

    double* x = (double*)malloc(n * sizeof(double));
    double* x_old = (double*)malloc(n * sizeof(double));
    if (x == NULL || x_old == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    for (i = 0; i < n; i++) {
        x[i] = 0;
        x_old[i] = 0;
    }

    int maxIterations;
    double tolerance;
    printf("Enter the maximum number of iterations: ");
    scanf("%d", &maxIterations);
    printf("Enter the tolerance: ");
    scanf("%lf", &tolerance);

    int iter = 0;
    double maxError;

    printf("Iter\t");
    for (i = 0; i < n; i++) {
        printf("x%d\t\t", i + 1);
    }
    printf("Error\n");

    do {
        maxError = 0;
        for (i = 0; i < n; i++) {
            x_old[i] = x[i];
        }

        for (i = 0; i < n; i++) {
            double sum = augmented.data[i][n];
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sum -= augmented.data[i][j] * x[j];
                }
            }
            x[i] = sum / augmented.data[i][i];

            double error = fabs(x[i] - x_old[i]);
            if (error > maxError) {
                maxError = error;
            }
        }

        printf("%d\t", iter);
        for (i = 0; i < n; i++) {
            printf("%lf\t", x[i]);
        }
        printf("%lf\n", maxError);

        iter++;
    } while (maxError > tolerance && iter < maxIterations);

    printf("The solution after %d iterations is:\n", iter);
    for (i = 0; i < n; i++) {
        printf("x%d = %lf\n", i + 1, x[i]);
    }

    free(x);
    free(x_old);
    freeMatrix(augmented);
}

void numericalDifferentiation() {
    double x, h;
    printf("Enter the point of differentiation (x): ");
    scanf("%lf", &x);
    printf("Enter the step size (h): ");
    scanf("%lf", &h);

    double forwardDifference = (evaluate_func(function_expr, x + h) - evaluate_func(function_expr, x)) / h;
    double backwardDifference = (evaluate_func(function_expr, x) - evaluate_func(function_expr, x - h)) / h;
    double centralDifference = (evaluate_func(function_expr, x + h) - evaluate_func(function_expr, x - h)) / (2 * h);

    printf("Forward Difference: %lf\n", forwardDifference);
    printf("Backward Difference: %lf\n", backwardDifference);
    printf("Central Difference: %lf\n", centralDifference);
}

void simpsonsRule() {
    double a, b;
    int i, n;
    printf("Enter the lower limit (a): ");
    scanf("%lf", &a);
    printf("Enter the upper limit (b): ");
    scanf("%lf", &b);
    printf("Enter the number of intervals (n): ");
    scanf("%d", &n);

    if (n % 2 != 0) {
        printf("Number of intervals (n) must be even for Simpson's rule.\n");
        return;
    }

    double h = (b - a) / n;
    double integral = evaluate_func(function_expr, a) + evaluate_func(function_expr, b);

    for (i = 1; i < n; i += 2) {
        integral += 4 * evaluate_func(function_expr, a + i * h);
    }
    for (i = 2; i < n; i += 2) {
        integral += 2 * evaluate_func(function_expr, a + i * h);
    }

    integral *= h / 3;
    printf("The integral is: %lf\n", integral);
}

void trapezoidalRule() {
    double a, b;
    int i, n;
    printf("Enter the lower limit (a): ");
    scanf("%lf", &a);
    printf("Enter the upper limit (b): ");
    scanf("%lf", &b);
    printf("Enter the number of intervals (n): ");
    scanf("%d", &n);

    double h = (b - a) / n;
    double integral = (evaluate_func(function_expr, a) + evaluate_func(function_expr, b)) / 2.0;

    for (i = 1; i < n; i++) {
        integral += evaluate_func(function_expr, a + i * h);
    }

    integral *= h;
    printf("The integral is: %lf\n", integral);
}

void gregoryNewtonInterpolation() {
    int i, j, n;
    printf("Enter the number of data points: ");
    scanf("%d", &n);

    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double** diff = (double**)malloc(n * sizeof(double*));
    if (x == NULL || y == NULL || diff == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    for (i = 0; i < n; i++) {
        diff[i] = (double*)malloc(n * sizeof(double));
        if (diff[i] == NULL) {
            printf("Memory allocation failed.\n");
            exit(1);
        }
    }

    printf("Enter the data points (x y):\n");
    for (i = 0; i < n; i++) {
        scanf("%lf %lf", &x[i], &y[i]);
        diff[i][0] = y[i];
    }

    for (j = 1; j < n; j++) {
        for (i = 0; i < n - j; i++) {
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1];
        }
    }

    double xp;
    printf("Enter the interpolation point: ");
    scanf("%lf", &xp);

    double yp = y[0];
    double term = 1;
    for (i = 1; i < n; i++) {
        term *= (xp - x[i - 1]);
        yp += (term * diff[0][i]) / factorial(i);
    }

    printf("The interpolated value at %lf is: %lf\n", xp, yp);

    free(x);
    free(y);
    for (i = 0; i < n; i++) {
        free(diff[i]);
    }
    free(diff);
}

unsigned long long factorial(int n) {
    int i;
    if (n < 0) {
        printf("Factorial is not defined for negative numbers.\n");
        return 0;
    }
    unsigned long long result = 1;
    for (i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}
