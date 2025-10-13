classdef MOEADBOS < ALGORITHM
    % <multi/many> <real/integer/label/binary/permutation>

    %------------------------------- Reference --------------------------------
    % Q. Fan, D. Yang, J. Peng, H. Li, J. Wang, and A. Yin Decomposition-based 
    % Multi-objective Evolutionary Algorithm for Bi-optimal Selection, Swarm and
    % Evolutionary Computation, 2025.
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);    
            W = 1./W./repmat(sum(1./W,2),1,size(W,2));            
            T  = ceil(Problem.N/10);                             
            nr = ceil(Problem.N/100);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);                                      
            [~,B] = sort(B,2);                                    
            B = B(:,1:T);                                         

            %% Generate random population
            NPC = Problem.Initialization(size(W,1));              
            Zmin   = min(NPC.objs,[],1);                          
            Zmax   = max(NPC.objs,[],1);                          
            [PC,nND] = EnvironmentalSelection(NPC,Problem.N);     

            %% Optimization
            while Algorithm.NotTerminated(PC)
                % PC evolving

                NewPC = Exploration(Problem,PC,NPC,nND,Problem.N);
                % NPC selection
                for i = 1 : length(NewPC)

                    % Update the ideal point
                    Zmin = min([Zmin;NewPC.objs],[],1);
                    Zmax  = max([Zmax;NewPC.objs],[],1);

                    % Update at most one solution in NPC
                    P     = randperm(length(NPC));
                    g_old = max((abs(NPC(P).objs-repmat(Zmin,length(P),1))./repmat(Zmax-Zmin,length(P),1)).*W(P,:),[],2);
                    g_new = max((repmat(abs(NewPC(i).obj-Zmin),length(P),1)./repmat(Zmax-Zmin,length(P),1)).*W(P,:),[],2);
                    NPC(P(find(g_old>=g_new,nr))) = NewPC(i); 
                end

                % NPC evolving
                NewNPC(1:length(NPC)) = SOLUTION();
                for i = 1 : length(NPC)

                % Choose the parents
                    if rand <= 0.8   
                        P1 = B(i,randperm(size(B,2)));
                        P2 = B(P1(5),randperm(size(B,2)));
                    else
                        P1 = randperm(length(NPC));
                        P2 = randperm(length(NPC));
                    end
                    if P1(1)==P2(1)
                        P2(1)= [];
                    end

                % Generate an offspring
                    if Problem.FE/Problem.maxFE <=0.5

                       NewNPC(i) = OperatorGAhalf(Problem,NPC([P1(1),P2(1)]));
                    else
                       NewNPC(i) = OperPSO(Problem,NPC(i),NPC(P1(1)),NPC(P2(1)));
                    end
                 % Update the ideal point
                    Zmin = min([Zmin;NewNPC.objs],[],1);
                    Zmax  = max([Zmax;NewNPC.objs],[],1);

                 % Update the solutions in P by modified Tchebycheff approach
                    g_old = max((abs(NPC(P1).objs-repmat(Zmin,length(P1),1))./repmat(Zmax-Zmin,length(P1),1)).*W(P1,:),[],2);
                    g_new = max((repmat(abs(NewNPC(i).obj-Zmin),length(P1),1)./repmat(Zmax-Zmin,length(P1),1)).*W(P1,:),[],2);
                    NPC(P1(find(g_old>=g_new,nr))) = NewNPC(i);
                end

                 % PC selection
                [PC,nND] = EnvironmentalSelection([PC,NewNPC,NewPC],Problem.N);
            end
        end
    end
end