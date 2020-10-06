#imports

using JuMP, Gurobi, DelimitedFiles, Random

function atualize_lowerBound(A,c,x_res,indCost)

    (m,n) = size(A)
    I = (1:m)
    J = (1:n)
    x_resLower = zeros(n,1)
    global x_res_temp = zeros(n,1)

    for indice_j in indCost
        indice_j = floor(Int,indice_j)
        if x_res[indice_j] == 1
            x_res_temp[indice_j] = 1
            value_mult = A*x_res_temp

            flag = 0
            for i in I
                if value_mult[i] > 1

                    flag = 1
                    #break
                end
            end

            if flag == 0
                x_resLower[indice_j] = 1
            end

            global x_res_temp = copy(x_resLower)

        end
    end

    return c'*x_resLower

end


function RelaxLagrang(A, c, maxiter = 50, tol = 1e-1)
    spp   = Model(Gurobi.Optimizer);
    (m,n) = size(A)

    I = (1:m)
    J = (1:n)

    @variable(spp, x[1:n], Bin);
    u     = 1*ones(m,1);
    zs = []
    irep    = 0;
    iter    = 0;
    lower_b = 0;
    upper_b = Inf;

    step_size = 0
    grad      = zeros(m,1);
    indCost   = zeros(1,n)

    for j1 in J
        maior_valor = -Inf
        for j2 in J
            if (c[j2] >=  maior_valor) && ((j2 in indCost) == false)
                global maior_indice = j2
                 maior_valor = c[j2]
            end
        end
        indCost[j1] = maior_indice
    end


    while iter <= maxiter && abs(upper_b - lower_b) >= tol

        iter += 1
        u = max.(0, u - step_size*grad)

        @objective(spp, Max, sum(c.*x) - sum(u[i]* (sum(A[i,j]*x[j] for j in (1:n)) - 1) for i in (1:m)))
        optimize!(spp)

        x_res    = value.(x)
        soma_res = A*x_res
        z        = objective_value(spp);
        append!(zs, z)

        best_ub_old = upper_b;
        upper_b = min(upper_b, z);

        if best_ub_old == upper_b
            irep += 1;
        else
            irep = 0;
            lower_b = atualize_lowerBound(A,c,x_res,indCost)[1];

        end

        grad = ones(m,1) - soma_res + ones(m,1)*rand(1)/10
        sum_g2 = sum(grad[i]^2 for i in I)
        #step_size = (z - lower_b)/sum_g2
        step_size = 1/iter
        #step_size  = 0.008

    end

    return zs, iter
end


function loadSPP(fname)
    f=open(fname)

    m, n = parse.(Int, split(readline(f)))
    C = parse.(Int, split(readline(f)))
    A=zeros(Int, m, n)
    for i=1:m
        readline(f)
        for valor in split(readline(f))
          j = parse(Int, valor)
          A[i,j]=1
        end
    end
    close(f)
    return C, A
end


function setSPP(solverSelected, C, A)
  m, n = size(A)
  ip = Model(solverSelected)
  @variable(ip, x[1:n], Bin)
  @objective(ip, Max, sum(C.*x))
  @constraint(ip , cte[i=1:m], sum(A[i,j] * x[j] for j=1:n) <= 1)
  return ip, x
end

function setSPP_relaxlin(solverSelected, C, A)
  m, n = size(A)
  ip = Model(solverSelected)
  @variable(ip, 0 <= x[1:n] <= 1)
  @objective(ip, Max, sum(C.*x))
  @constraint(ip , cte[i=1:m], sum(A[i,j] * x[j] for j=1:n) <= 1)
  return ip, x
end


function integer_test(x)
    if floor(x)^2 - x^2 == 0
        return 1
    end
    return 0
end


function branch_ceil(solverSelected, C, A, idxs = [])
  m, n = size(A)
  ip = Model(solverSelected)
  @variable(ip, 0 <= x[1:n] <= 1)
  for idx in idxs
    @constraint(ip, x[idx] == 1)
  end
  @objective(ip, Max, sum(C.*x))
  @constraint(ip , cte[i=1:m], sum(A[i,j] * x[j] for j=1:n) <= 1)
  return ip, x
end

function branch_floor(solverSelected, C, A, idxs = [])
  m, n = size(A)
  ip = Model(solverSelected)
  @variable(ip, 0 <= x[1:n] <= 1)
  for idx in idxs
    @constraint(ip, x[idx] == 0)
  end
  @objective(ip, Max, sum(C.*x))
  @constraint(ip , cte[i=1:m], sum(A[i,j] * x[j] for j=1:n) <= 1)
  return ip, x
end

function BranchNBound(A, c, maxiter = 50, tol = 1e-1)

    rp, rp_x = setSPP_relaxlin(Gurobi.Optimizer, c, A)
    nodes    = [[rp,[]]]
    s,       = size(nodes)
    values   = []
    idxs     = []

    while s >= 0


        if s == 0
            break
        end

        p,idxs   = nodes[1]

        if s == 1
            nodes = []
            s     = 0
        else
            nodes = nodes[2:s]
            s,     = size(nodes)
        end


        try
           optimize!(p)
        catch
           println(" Modelo Infeasible. Cortem as árvores! ")
           ss,       = size(idxs)
           idxs      = idxs[1:ss-1]

        end
        append!(values, objective_value(p))

        for (i,xs) in enumerate(value.((all_variables(p))))
            if integer_test(xs) == 0

                push!(idxs, i)
                print("----------> ", idxs, " <---------")
                print(" BRANCHING ")

                floor_prob, f_x = branch_floor(Gurobi.Optimizer, c, A,idxs)
                push!(nodes, floor_prob)

                ceil_prob, c_x  =  branch_ceil(Gurobi.Optimizer, c, A,idxs)
                push!(nodes, ceil_prob)

                s += 2
                print("--> ", values, " <---")
                break
            end
        end
    end

    return objective_value(rp), value.(rp_x), values
end


function Branch_Bound(A,c)

    rp, rp_x = setSPP(Gurobi.Optimizer, c, A)

    println("Solving...");
    set_optimizer_attribute(rp, "Presolve", 0)
    set_optimizer_attribute(rp, "Cuts", 0)
    set_optimizer_attribute(rp, "Heuristics", 0)
    optimize!(rp)
end


function Branch_Cut(A,c)

    p, p_x = setSPP(Gurobi.Optimizer, c, A)

    println("Solving...");
    set_optimizer_attribute(p, "Presolve", 0)
    set_optimizer_attribute(p, "Cuts", 0)
    set_optimizer_attribute(p, "Heuristics", 0)
    set_optimizer_attribute(p, "CliqueCuts", 1)
    optimize!(p)

end
