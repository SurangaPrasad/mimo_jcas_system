function Psi_out = solve_psi_cvx(N, T, Bd, Abar_grid, PBS)
    % N, T, Bd, Abar_grid, PBS are inputs from Python
    
    try
        cvx_begin quiet
            variable Psi(N,N) hermitian semidefinite
            variable alpha_cvx
            expression err(T)
            
            for tt = 1:T
                a_t = Abar_grid(:,tt);
                err(tt) = alpha_cvx * Bd(tt) - a_t' * Psi * a_t;
            end
            
            minimize( sum_square_abs(err) )
            subject to
                diag(Psi) == (PBS/N) * ones(N,1);
                alpha_cvx >= 0;
        cvx_end
        
        % Return the optimized matrix (or a fallback if CVX failed but didn't crash)
        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            Psi_out = Psi;
        else
            warning('CVX did not find an optimal solution. Using fallback.');
            Psi_out = (PBS/N) * eye(N);
        end
        
    catch ME
        warning('CVX toolbox error occurred. Using fallback.');
        disp(ME.message);
        Psi_out = (PBS/N) * eye(N);
    end
end