#ifndef ReachableSet_h
#define ReachableSet_h

//
//  Created by FMLAB4 on 28.11.17.
//  Copyright © 2017 FMLAB4. All rights reserved.
//

#define PROFIL_VNODE
#define MAXORDER 50

#include <iostream>
#include "vnode.h"
#include <ostream>
#include "fadiff.h"
#include "badiff.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include "UsingGnuplot.h"

#include "TicToc.hh"

extern Eigen::VectorXd L_hat_previous;  // abstraction.hh: compute_gb2

namespace mstom {
    
    template<class T>
    T VXdToV(T& v, Eigen::VectorXd vxd){
        for(int i=0;i<vxd.rows();i++)
            v.push_back(vxd(i));
        return v;
    }
    
    vector<double> VXdToV(Eigen::VectorXd vxd){    //Eigen to C++ vector
        vector<double> v;
        for(int i=0;i<vxd.rows();i++)
            v.push_back(vxd(i));
        return v;
    }
    
    class zonotope {
    public:
        int n;
        Eigen::MatrixXd generators; //Each column represents a generator
        Eigen::VectorXd centre;
        zonotope (){};
        zonotope (const Eigen::VectorXd& c, const Eigen::VectorXd& eta){
            centre = c;
            int dim = (int) eta.rows();
            generators = Eigen::MatrixXd::Zero(dim,dim);
            for (int i=0; i<dim; i++){
                generators(i,i) = eta(i)/2;
            }
        };
        
        template<class state_type>
        zonotope (state_type ct, state_type etat, int dim){
            Eigen::VectorXd c(dim), eta(dim);
            for(int i=0;i<dim;i++)
            {
                c(i) = ct[i];
                eta(i) = etat[i];
            }
            centre = c;
            generators = Eigen::MatrixXd::Zero(dim,dim);
            for (int i=0; i<dim; i++){
                generators(i,i) = eta(i)/2;
            }
        };
        
        
        //Minkowskii sum
        zonotope operator + (const zonotope& Zb){
            zonotope Ztemp;
            Ztemp.centre = centre + Zb.centre;
            Ztemp.generators = Eigen::MatrixXd::Zero(generators.rows(), generators.cols() + Zb.generators.cols());
            Ztemp.generators.block(0,0,generators.rows(),generators.cols()) = generators;
            Ztemp.generators.block(0,generators.cols(),Zb.generators.rows(),Zb.generators.cols()) = Zb.generators;
            return Ztemp;
        }
        
        zonotope operator + (Eigen::VectorXd& V){
            // zonotope + vector
            zonotope Ztemp;
            Ztemp.centre = centre + V;
            Ztemp.generators = generators;
            return Ztemp;
        }
        
        zonotope operator - (Eigen::VectorXd& V){
            // zonotope - vector
            zonotope Ztemp;
            Ztemp.centre = centre - V;
            Ztemp.generators = generators;
            return Ztemp;
        }
        
        zonotope operator - (const zonotope& Zb){
            zonotope Ztemp;
            Ztemp.centre = centre - Zb.centre;
            Ztemp.generators = Eigen::MatrixXd::Zero(generators.rows(), generators.cols() + Zb.generators.cols());
            Ztemp.generators.block(0,0,generators.rows(),generators.cols()) = generators;
            Ztemp.generators.block(0,generators.cols(),Zb.generators.rows(),Zb.generators.cols()) = (-1 * Zb.generators);
            return Ztemp;
        }
        
        zonotope operator * (double a){
            // zonotope * scalar
            zonotope Ztemp;
            Ztemp.centre = centre * a;
            Ztemp.generators = generators * a;
            return Ztemp;
        }
         
        void display(){
            int r = generators.rows();
            int gc = generators.cols();
            Eigen::MatrixXd M(r,1+gc);
            M.block(0,0,r,1) = centre;
            M.block(0,1,r,gc) = generators;
            std::cout<< "\nzonotope: \n"<< M << std::endl;        }
        
    };
    
    // product of matrix with Zonotope
    zonotope operator * (const Eigen::MatrixXd& M, const zonotope& Z){
        zonotope Ztemp;
        Ztemp.centre = M * Z.centre;
        Ztemp.generators = M * Z.generators;
        return Ztemp;
    }
    
    // vector<interval> to zonotope
    mstom::zonotope vecIntToZono(vector<interval> Kprime){
        int dim = Kprime.size();
        Eigen::VectorXd c(dim), lb(dim), ub(dim);
        for(int i=0;i<dim;i++)
        {
            lb(i) = inf(Kprime[i]);
            ub(i) = sup(Kprime[i]);
        }
        c = (lb + ub) * 0.5;
        mstom:: zonotope Z(c, ub-lb);
        return Z;
    }
    
    class intervalMatrix{
    public:
        Eigen::MatrixXd lb;
        Eigen::MatrixXd ub;
        intervalMatrix(){};
        zonotope operator * ( const zonotope& Z){    //interval matrix map for zonotope
            int row = (int) Z.generators.rows();
            int col = (int) Z.generators.cols();
            zonotope Ztemp;
            Eigen::MatrixXd M_tilde = (lb + ub)/2;
            Eigen::MatrixXd M_hat = ub - M_tilde;
            Ztemp.centre = M_tilde * Z.centre;
            Ztemp.generators = Eigen::MatrixXd::Zero(row, col+row);
            Ztemp.generators.block(0,0,row,col) = M_tilde * Z.generators;
            Eigen::MatrixXd v = Eigen::MatrixXd::Zero(row,row);
            for (int i=0; i<row; i++){
                v(i,i) = M_hat.row(i) * (Z.centre.cwiseAbs() + Z.generators.cwiseAbs().rowwise().sum() ) ;
            }
            Ztemp.generators.block(0,col,row,row) = v;
            return Ztemp;
        }
        
        intervalMatrix operator * (double a){
            intervalMatrix Mitemp;
            Eigen::MatrixXd temp1 = lb*a;
            Eigen::MatrixXd temp2 = ub*a;
            Mitemp.lb = temp1.array().min(temp2.array()).matrix();  //converted to array for min, then reconverted to matrix
            Mitemp.ub = temp1.array().max(temp2.array()).matrix();
            return Mitemp;
        }
        
        intervalMatrix operator + (const Eigen::MatrixXd& M){
            intervalMatrix Mitemp;
            Mitemp.lb = lb + M;
            Mitemp.ub = ub + M;
            return Mitemp;
        }
        
        intervalMatrix operator + (const intervalMatrix& Mi){
            intervalMatrix Mitemp;
            Mitemp.lb = lb + Mi.lb;
            Mitemp.ub = ub + Mi.ub;
            return Mitemp;
        }
        
    };
    
    zonotope convexHull(const zonotope& Z1, const Eigen::MatrixXd& eAr){
        // requires initial zonotope and exponential matirx (state-transition matrix)
        int r1 = (int) Z1.generators.rows();
        int c1 = (int) Z1.generators.cols();
        zonotope Ztemp;
        Ztemp.centre = (Z1.centre + eAr * Z1.centre)/2;
        Ztemp.generators = Eigen::MatrixXd::Zero(r1, 1+2*c1);
        Ztemp.generators.block(0,0,r1,c1) = (Z1.generators + eAr * Z1.generators)/2;
        Ztemp.generators.block(0,c1,r1,1) = (Z1.centre - eAr * Z1.centre)/2;
        Ztemp.generators.block(0,1+c1,r1,c1) = (Z1.generators - eAr * Z1.generators)/2;
        return Ztemp;
    }
    
    zonotope convexHull(const zonotope& Z1, const zonotope& Z2){
        // when zonotopes may have different number of generators
        zonotope Ztemp;
        int r = (int)Z1.generators.rows();
        int c1 = (int)Z1.generators.cols();
        int c2 = (int)Z2.generators.cols();
        int c = (c1 > c2 ? c1 : c2);
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(r,c);
        Ztemp.generators = Eigen::MatrixXd::Zero(r,1+2*c);
        if(c1<c2)
        {
            M.block(0,0,r,c1) = Z1.generators;
            Ztemp.generators.block(0,0,r,c) = 0.5 * (M + Z2.generators);
            Ztemp.generators.block(0,c,r,1) = 0.5 * (Z1.centre - Z2.centre);
            Ztemp.generators.block(0,c+1,r,c) = 0.5 * (M - Z2.generators);
        }
        else
        {
            M.block(0,0,r,c2) = Z2.generators;
            Ztemp.generators.block(0,0,r,c) = 0.5 * (M + Z1.generators);
            Ztemp.generators.block(0,c,r,1) = 0.5 * (Z2.centre - Z1.centre);
            Ztemp.generators.block(0,c+1,r,c) = 0.5 * (M - Z1.generators);
        }
        Ztemp.centre = 0.5*(Z1.centre + Z2.centre);
        return Ztemp;
    }
    
    
    zonotope convexHull(vector<zonotope>& stora){
        zonotope Z = stora[0];
        for(int i=1;i<stora.size();i++)
        {
            Z = convexHull(Z, stora[i]);
        }
        
        return Z;
    }
    
    
    
    double factorial(double n){
        return (n==1 || n==0) ? 1 : factorial(n-1) * n;
    }
    
    double compute_epsilon(const Eigen::MatrixXd& A, const double& r, int& p){
        double norm_rA = (r*A).cwiseAbs().rowwise().sum().maxCoeff();    
        double epsilone = norm_rA/(p+2);    
        while (epsilone >= 1){
            p += 1;
            epsilone = norm_rA/(p+2);
         }
        
        
        return epsilone;
    }
    
    double p_adjust_Er_bound(Eigen::MatrixXd& A, double& r, int& p, double& epsilone){
        // adjusts p, epsilone
        Eigen::MatrixXd rA = r * A;
        double norm_rA = rA.cwiseAbs().rowwise().sum().maxCoeff();
        double temp = pow(norm_rA,p+1) / factorial(p+1);
        double bound = temp / (1-epsilone);
        while(bound > pow(10,-5))  
        {
            p++;
            epsilone = norm_rA/(p+2);
            temp = temp * norm_rA / (p+1);
            bound = temp / (1-epsilone);
        }
        return bound;
    }
    
    void matrix_product(double* M1, double* M2, double* Mresult, int m, int n, int q){
        // size M1 = mxn; size M2 = nxq; M1xM2
        double temp;
        for(int i=0;i<m;i++)
            for(int j=0;j<q;j++)
            {
                temp = 0;
                for(int k=0;k<n;k++)
                {
                    temp += M1[i*n+k] * M2[k*q+j];
                }
                Mresult[i*q+j] = temp;
            }
    }
    
    void sum_matrix(double M1[], double M2[], int m, int n){
        // result stored in M1; size = m x n
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
                M1[i*n+j] = M1[i*n+j] + M2[i*n+j];
    }
    
    intervalMatrix matrix_exponential(Eigen::MatrixXd& A, double r, int& p, double& epsilone, intervalMatrix& Er, double Ar_powers_fac[], double bound){
        // (p+1) terms truncation; result is an Interval Matrix
        // p, epsilone: may be altered
        int state_dim = A.rows();
        intervalMatrix Mitemp;
        Mitemp.lb = Eigen::MatrixXd::Constant(state_dim,state_dim,-1);
        Mitemp.ub = Eigen::MatrixXd::Constant(state_dim,state_dim,1);
        
        Mitemp = Mitemp * bound; // E(r)
        Er = Mitemp;
        double Btemp[state_dim*state_dim];
        double Btemp2[state_dim*state_dim] ;
        double Btemp3[state_dim*state_dim];
        double* BtempPointer2;
        double* BtempPointer3;
        double* swapPointer;
        double fac = 1; // factorial(1)
        for(int i=0;i<state_dim;i++)
            for(int j=0;j<state_dim;j++)
            {
                Btemp[i*state_dim+j] = 0;  // initialise Btemp to 0
                Btemp2[i*state_dim+j] = r * A(i,j); //put Ar in Btemp2
                Ar_powers_fac[state_dim*state_dim + i*state_dim + j] = r * A(i,j);  // store rA as the second matrix in Ar_powers_fac[]
                if(i == j)   // identity matrix as the first matrix in Ar_powers_fac[]
                    
                    Ar_powers_fac[i*state_dim+j] = 1;
                else
                    Ar_powers_fac[i*state_dim+j] = 0;
            }
        sum_matrix(Btemp,Ar_powers_fac, state_dim, state_dim);   // put identity matrix in Btemp
        sum_matrix(Btemp, &Ar_powers_fac[state_dim*state_dim], state_dim, state_dim);// add Ar to Btemp
        BtempPointer2 = Btemp2;
        BtempPointer3 = Btemp3;
        for (int i=2; i<=p; i++)
        {
             
            matrix_product(&Ar_powers_fac[state_dim*state_dim], BtempPointer2, BtempPointer3, state_dim, state_dim, state_dim); // pow(rA,i) in Btemp3
            fac = fac * i;
            swapPointer = BtempPointer2;
            BtempPointer2 = BtempPointer3;// pow(rA,i)
            BtempPointer3 = swapPointer;
            for(int ii=0;ii<state_dim;ii++)
                for(int jj=0;jj<state_dim;jj++)
                {
                    Ar_powers_fac[i*state_dim*state_dim + ii*state_dim + jj] = BtempPointer2[ii*state_dim+jj] / fac;
                    Btemp[ii*state_dim + jj] = Btemp[ii*state_dim + jj] + Ar_powers_fac[i*state_dim*state_dim + ii*state_dim + jj];
                }
        }
         for(int i=0;i<state_dim;i++)
            for(int j=0;j<state_dim;j++)
            {
                Mitemp.lb(i,j) = Mitemp.lb(i,j) + Btemp[i*state_dim + j];
                Mitemp.ub(i,j) = Mitemp.ub(i,j) + Btemp[i*state_dim + j];
            }
        return Mitemp;
    }
    
    intervalMatrix compute_F(const int& p, const double& r, const Eigen::MatrixXd& A, const intervalMatrix& Er, double Ar_powers_fac[]){
        int state_dim = A.rows();
        intervalMatrix Ftemp;
        Ftemp.ub = Eigen::MatrixXd::Zero(state_dim, state_dim);
        Ftemp.lb = Eigen::MatrixXd::Zero(state_dim, state_dim);
        Eigen::MatrixXd temp(state_dim,state_dim);
        for (int i=2; i<=p; i++){
              double data = (pow(i,-i/(i-1)) - pow(i,-1/(i-1)));
            for(int i2=0;i2<state_dim;i2++)
                for(int j=0;j<state_dim;j++)
                    temp(i2,j) = data * Ar_powers_fac[i*state_dim*state_dim + i2*state_dim + j];
            
            for(int i1=0; i1<A.rows(); i1++){
                for(int i2=0; i2<A.rows(); i2++){
                    if (temp(i1,i2) < 0)
                        Ftemp.lb(i1,i2) += temp(i1,i2);
                    else
                        Ftemp.ub(i1,i2) += temp(i1,i2);
                }
            }
        }
        Ftemp = Ftemp + Er;
        return Ftemp;
    }
    
    intervalMatrix compute_F_tilde(const int& p, const double& r, Eigen::MatrixXd& A, intervalMatrix& Er, int isOriginContained, double Ar_powers_fac[]){
        int state_dim = A.rows();
        intervalMatrix Ftemp;
        Ftemp.ub = Eigen::MatrixXd::Zero(state_dim,state_dim);
        Ftemp.lb = Eigen::MatrixXd::Zero(state_dim,state_dim);
        if(!isOriginContained)
        {
            Eigen::MatrixXd temp(state_dim,state_dim);
            for (int i=2; i<=p; i++){
                   double data = (pow(i,-i/(i-1)) - pow(i,-1/(i-1)));
                for(int i2=0;i2<state_dim;i2++)
                    for(int j=0;j<state_dim;j++)
                        temp(i2,j) = data * Ar_powers_fac[(i-1)*state_dim*state_dim + i2*state_dim + j] * r / i;
                
                for(int i1=0; i1<A.rows(); i1++){
                    for(int i2=0; i2<A.rows(); i2++){
                        if (temp(i1,i2) < 0)
                            Ftemp.lb(i1,i2) += temp(i1,i2);
                        else
                            Ftemp.ub(i1,i2) += temp(i1,i2);
                    }
                }
            }
            Eigen::VectorXd temp2 = A.cwiseAbs().rowwise().sum();  
            Ftemp = Ftemp + Er * pow(temp2.maxCoeff(),-1);
        }
         return Ftemp;
    }
    
    intervalMatrix compute_Data_interm(intervalMatrix Er, double r, int p, Eigen::MatrixXd& A, double Ar_powers_fac[]){
        int state_dim = A.rows();
        Eigen::MatrixXd Mtemp = Eigen::MatrixXd::Zero(state_dim,state_dim);
        for (int i=0; i<=p; i++){
            for(int i2=0;i2<state_dim;i2++)
                for(int j=0;j<state_dim;j++)
                    Mtemp(i2,j) = Mtemp(i2,j) + Ar_powers_fac[i*state_dim*state_dim + i2*state_dim + j] * r / (i+1);
        }
        intervalMatrix Mintv = Er;
        Mintv = Mintv * r;  // r*E(r)
        Mintv = Mintv + Mtemp;  // (Aˆ-1)*(exp(Ar) - I)
        return Mintv;
    }
    
    intervalMatrix IntervalHull(const zonotope& Z)
    {
        intervalMatrix iM;
        iM.lb = Z.centre - Z.generators.cwiseAbs().rowwise().sum();
        iM.ub = Z.centre + Z.generators.cwiseAbs().rowwise().sum();
        return iM;
    }
    
    zonotope project(zonotope Z, int a, int b){
        // project on to the dimensions a and b
        zonotope Zp;
        Eigen::Vector2d c;
        Eigen::MatrixXd M(2,Z.generators.cols());
        c(0) = Z.centre(a-1);
        c(1) = Z.centre(b-1);
        M.row(0) = Z.generators.row(a-1);
        M.row(1) = Z.generators.row(b-1);
        Zp.centre = c;
        Zp.generators = M;
        return Zp;
    }
    
    template <typename T>
    vector<size_t> sort_indexes(const vector<T> &v) {
        
        // initialize original index locations
        vector<size_t> idx(v.size());
        iota(idx.begin(), idx.end(), 0);
        
        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),
             [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
        
        return idx;
    }
    
    zonotope deletezeros(zonotope Z){
        // delete zero generators
        zonotope Zd;
        Eigen::MatrixXd vtemp = Z.generators.cwiseAbs().colwise().sum();
        int ncol = (vtemp.array() == 0).select(vtemp, 1).sum();
        Eigen::MatrixXd M(Z.generators.rows(),ncol);
        int index = 0;
        for(int i=0;i<vtemp.cols();i++)
        {
            if(vtemp(i) != 0)
            {
                M.col(index) = Z.generators.col(i);
                index++;
            }
        }
        Zd.centre = Z.centre;
        Zd.generators = M;
        return Zd;
    }
    
    std::vector<std::pair<double, double>> vertices(zonotope Z){
        std::vector<std::pair<double, double>> v;
        Eigen::VectorXd c = Z.centre;
        Eigen::MatrixXd g = Z.generators;
        int n = g.cols();   // number of generators
        double xmax = g.row(0).cwiseAbs().sum();
        double ymax = g.row(1).cwiseAbs().sum();
        Eigen::MatrixXd gup = g;
        for(int i=0;i<g.cols();i++)
        {
            if(g(1,i)<0)    // if 2nd column of g is negative, then reverse the vector to make it point up
                gup.col(i) = -1 * g.col(i);
        }
        std::vector<double> angles;
        for(int i =0;i<n;i++)
        {
            double angletemp = atan2(gup(1,i), gup(0,i));
            if(angletemp<0)
                angletemp += 2 * M_PI;
            angles.push_back(angletemp);
        }
        std::vector<size_t> sortIndexes = sort_indexes(angles);
        Eigen::MatrixXd p = Eigen::MatrixXd::Zero(2, n+1);
        for(int i=0;i<n;i++)
        {
            p.col(i+1) = p.col(i) + 2 * gup.col(sortIndexes[i]);
        }
        p.row(0) = p.row(0) + Eigen::MatrixXd::Constant(1,p.cols(), (xmax - p.row(0).maxCoeff()));
        p.row(1) = p.row(1) - Eigen::MatrixXd::Constant(1,p.cols(), ymax);
        Eigen::MatrixXd p2(2, (p.cols()*2));
        p2.row(0).head(p.cols()) = p.row(0);
        p2.row(0).tail(p.cols()) = Eigen::MatrixXd::Constant(1,p.cols(), (p.row(0).tail(1) + p.row(0).head(1))(0)) - p.row(0);
        p2.row(1).head(p.cols()) = p.row(1);
        p2.row(1).tail(p.cols()) = Eigen::MatrixXd::Constant(1,p.cols(), (p.row(1).tail(1)+p.row(1).head(1))(0)) - p.row(1);
        p2.row(0) += Eigen::MatrixXd::Constant(1,p2.cols(), c(0));
        p2.row(1) += Eigen::MatrixXd::Constant(1,p2.cols(), c(1));
        for(int i=0;i<p2.cols();i++)
            v.push_back(std::make_pair(p2(0,i),p2(1,i)));
        
        return v;
    }
    
    Eigen::MatrixXd verticesH(zonotope Z){
        // vertices for H-representation; same as vertices(), difference is only in the return type
        Eigen::VectorXd c = Z.centre;
        Eigen::MatrixXd g = Z.generators;
        int n = g.cols();   // number of generators
        double xmax = g.row(0).cwiseAbs().sum();
        double ymax = g.row(1).cwiseAbs().sum();
        Eigen::MatrixXd gup = g;
        for(int i=0;i<g.cols();i++)
        {
            if(g(1,i)<0)    // if 2nd column of g is negative, then reverse the vector to make it point up
                gup.col(i) = -1 * g.col(i);
        }
        std::vector<double> angles;
        for(int i =0;i<n;i++)
        {
            double angletemp = atan2(gup(1,i), gup(0,i));
            if(angletemp<0)
                angletemp += 2 * M_PI;
            angles.push_back(angletemp);
        }
        std::vector<size_t> sortIndexes = sort_indexes(angles);
        Eigen::MatrixXd p = Eigen::MatrixXd::Zero(2, n+1);
        for(int i=0;i<n;i++)
        {
            p.col(i+1) = p.col(i) + 2 * gup.col(sortIndexes[i]);
        }
        p.row(0) = p.row(0) + Eigen::MatrixXd::Constant(1,p.cols(), (xmax - p.row(0).maxCoeff()));
        p.row(1) = p.row(1) - Eigen::MatrixXd::Constant(1,p.cols(), ymax);
        Eigen::MatrixXd p2(2, (p.cols()*2));
        p2.row(0).head(p.cols()) = p.row(0);
        p2.row(0).tail(p.cols()) = Eigen::MatrixXd::Constant(1,p.cols(), (p.row(0).tail(1) + p.row(0).head(1))(0)) - p.row(0);
        p2.row(1).head(p.cols()) = p.row(1);
        p2.row(1).tail(p.cols()) = Eigen::MatrixXd::Constant(1,p.cols(), (p.row(1).tail(1)+p.row(1).head(1))(0)) - p.row(1);
        p2.row(0) += Eigen::MatrixXd::Constant(1,p2.cols(), c(0));
        p2.row(1) += Eigen::MatrixXd::Constant(1,p2.cols(), c(1));
        // vertex in column
        return p2;
    }
    
    void H_rep(zonotope& Z, Eigen::VectorXd& M){
        // H representation for 2D zonotope
        if(Z.centre.rows() > 2)
            cout << "Please use a 2D zonotope\n";
        Eigen::MatrixXd v = verticesH(Z);
        int n = v.cols();   // number of vertices
        Eigen::MatrixXd temp(v.rows(),v.cols());
        temp.block(0,0,v.rows(),v.cols()-1) = v.block(0,1,v.rows(),v.cols()-1);
        temp.col(v.cols()-1) = v.col(0);
        temp = temp - v;    // column as edge vector
        Eigen::Matrix2d temp2;
        temp2 << 0, 1, -1, 0;
        temp = temp2 * temp;    // perpendicular vectors
        Eigen::VectorXd Ma(n);
        for(int i=0;i<n;i++)
        {
            double dd = temp.col(i).transpose() * v.col(i);
            int j = ((i+n/2)< n)? (i+n/2) : (i+n/2-n);
            if(temp.col(i).transpose() * v.col(j) < dd)
            {
                  Ma(i) = dd;
            }
            else
            {
                  Ma(i) = -dd;
            }
        }
        M = Ma;
    }
    
    bool isOriginInZonotope(zonotope& Z){
        // only for 2D zonotopes
        Eigen::VectorXd M;
        H_rep(Z,M); // H representation; last row of M stores d; h.x < d inside the Z
        // for origin only sign of d need to be checked
        bool chk = (M.minCoeff()>=0);
        return chk;
    }
    
    void plot(zonotope Z, int a1, int a2){
        Gnuplot gp;
        //gp << "set terminal lua\n";
        std::vector<std::pair<double, double>> v = vertices(project(Z, a1, a2));
        //gp << "set output 'my_graph_1.png'\n";
        //gp << "set xrange [-2:2]\nset yrange [-2:2]\n";
        gp << "plot '-' with lines title 'cubic'\n";
        gp.send1d(v);
    }
    
    void plot(std::vector<zonotope> Zv, int a1, int a2){
        // a1, a2 : dimensions to plot
        Gnuplot gp;
        gp << "set grid\n";
        //gp << "set output 'my_graph_1.png'\n";
        for(int i=0;i<Zv.size();i++)
        {
            if(i==0)
                gp << "plot '-' with lines title 'cubic'";
            else
                gp << "'-' with lines";
            if(i==Zv.size()-1)
                gp << "\n";
            else
                gp << ", ";
        }
        for(int i=0;i<Zv.size();i++)
        {
            std::vector<std::pair<double, double>> v = vertices(project(Zv[i], a1, a2));
            gp.send1d(v);
        }
    }
    
    void plot(std::vector<zonotope> Zv, int a1, int a2, bool tb){
        // a1, a2 : dimensions to plot
        Gnuplot gp;
        gp << "set grid\n";
        
        //gp << "set output 'my_graph_1.png'\n";
        for(int i=0;i<Zv.size();i++)
        {
            if(i==0)
                gp << "plot '-' with lines title " << ((tb)? "'True'":"'False'");
            else
                gp << "'-' with lines";
            if(i==Zv.size()-1)
                gp << "\n";
            else
                gp << ", ";
        }
        for(int i=0;i<Zv.size();i++)
        {
            std::vector<std::pair<double, double>> v = vertices(project(Zv[i], a1, a2));
            gp.send1d(v);
        }
    }
    
    void plot(std::vector<double> L){
        Gnuplot gp;
        gp << "set output 'my_graph.png'\n";
        gp << "plot '-' with points\n";
        gp.send1d(L);
    }
    
    void plotstore(std::vector<zonotope>& PlotStorage, zonotope Z){
        PlotStorage.push_back(Z);
    }
    
    void plotstore(std::vector<zonotope>& PlotStorage, std::vector<zonotope> Zv){
        for(int i=0;i<Zv.size();i++)
        {
            PlotStorage.push_back(Zv[i]);
        }
    }
    
    void printVector(std::vector<double> v){
        std::cout<< "The vector is: \n";
        for(int i=0;i<v.size();i++)
            std::cout << v[i] << ", ";
        std::cout << std::endl;
    }
    
    
} // end namespace mstom

//#############################################################################
// Derivative Hessian


template<typename Tx, typename Tu>
Tx funcLj_system(Tx x, Tu u, Tx xx);


void computeJacobian(double A[], const Eigen::VectorXd& x_bar, Eigen::VectorXd uin)
{
    int dim = x_bar.rows();
    B<double> x[dim], xx[dim], *f;
    for(int i=0;i<dim;i++)
        x[i]=x_bar(i);
    f = funcLj_system(x, uin, xx);
    for(int i=0;i<dim;i++)
        f[i].diff(i,dim);
    for(int j=0;j<dim;j++)
        for(int i=0;i<dim;i++)
            A[j*dim+i] = x[i].d(j);
}

template<typename Fu>
void computeJacobian_Lu_array(vnodelp::interval xin[], Fu u[], Eigen::MatrixXd& L,const int dim){
    // from function in main file; using arrays
    // jacobian for L(u) for growth bound
    B<vnodelp::interval>* f;
    B<vnodelp::interval> x[dim];
    B<vnodelp::interval> xx[dim];
    for(int i=0;i<dim;i++)
        x[i] = xin[i];
    f = funcLj_system(x, u, xx);  // evaluate function and derivatives
    for(int i=0;i<dim;i++)
        f[i].diff(i,dim);
    for(int j=0;j<dim;j++)
        for(int i =0;i<dim;i++)
        {
            if(j==i)
                L(j,i) = vnodelp::sup(x[i].d(j));
            else
                L(j,i) = vnodelp::mag(x[i].d(j));
        }
}

template<typename Fu>
void computeJacobian_Lu_array2(vnodelp::interval xin[], Fu u[], const int dim, int jin, double* LuStore){
    // without eigen::matrix
    // from function in main file; using arrays
    // jacobian for L(u) for growth bound
    B<vnodelp::interval>* f;
    B<vnodelp::interval> x[dim];
    B<vnodelp::interval> xx[dim];
    for(int i=0;i<dim;i++)
        x[i] = xin[i];
    f = funcLj_system(x, u, xx);  // evaluate function and derivatives
    for(int i=0;i<dim;i++)
        f[i].diff(i,dim);
    for(int j=0;j<dim;j++)
        for(int i =0;i<dim;i++)
        {
            if(j==i)
                LuStore[jin*dim*dim+j*dim+i] = vnodelp::sup(x[i].d(j));
            else
                LuStore[jin*dim*dim+j*dim+i] = vnodelp::mag(x[i].d(j));
        }
}

template<typename T2>
void compute_J_abs_max(const mstom::intervalMatrix& iM, Eigen::MatrixXd J_abs_max[], T2 u)
{
    int nm = iM.lb.rows(); 
    vnodelp::interval d2f;
    B<F<vnodelp::interval>>* f;
    B<F<vnodelp::interval>> x[nm], xx[nm];
    for(int j=0;j<nm;j++)
    {
        x[j] = vnodelp::interval(iM.lb(j,0), iM.ub(j,0));
        x[j].x().diff(j,nm);
    }
    f = funcLj_system(x, u, xx);
    for(int i=0;i<nm;i++)
        f[i].diff(i,nm);
    
    Eigen::MatrixXd M(nm,nm);
    for(int i=0;i<nm;i++)
    {
        for(int j=0;j<nm;j++)
            for(int k=0;k<nm;k++)
            {
                d2f = x[j].d(i).d(k);
                M(j,k) = vnodelp::mag(d2f);
              }
        J_abs_max[i] = M;
    }
}

void compute_H(const mstom::intervalMatrix& iM, std::vector<vnodelp::iMatrix>& H, Eigen::VectorXd uin)
{
    int nm = iM.lb.rows(); // state_dimension
    vnodelp::interval d2f;
    B<F<vnodelp::interval>>* f;
    B<F<vnodelp::interval>> x[nm], xx[nm];
     for(int j=0;j<nm;j++)
    {
        x[j] = vnodelp::interval(iM.lb(j,0), iM.ub(j,0));
        x[j].x().diff(j,nm);
    }
    f = funcLj_system(x, uin, xx);
    for(int i=0;i<nm;i++)
        f[i].diff(i,nm);
    
    vnodelp::iMatrix M;
    vnodelp::sizeM(M,nm);
    for(int i=0;i<nm;i++)
    {
        for(int j=0;j<nm;j++)
            for(int k=0;k<nm;k++)
            {
                d2f = x[j].d(i).d(k);
                M[j][k] = (d2f);
             }
        H.push_back(M);
    }
}

Eigen::MatrixXd pMatrix_to_MatrixXd(vnodelp::pMatrix pM)
{
    int n = pM.size();
    Eigen::MatrixXd M(n,n);
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            M(i,j) = pM[i][j];
        }
    }
    return M;
}

Eigen::VectorXd pV_to_V(vnodelp::pVector pV)
{
    int n = pV.size();
    Eigen::VectorXd V(n);
    for(int i=0;i<n;i++)
        V(i) = pV[i];
    return V;
}

mstom::zonotope compute_quad(mstom::zonotope Z, std::vector<Eigen::MatrixXd> H_mid)
{
    int row, gcol; // gcol = number of generators
    row = Z.centre.rows();
    gcol = Z.generators.cols();
    mstom::zonotope Zq;
    Eigen::MatrixXd Zmat(row, 1+gcol);
    Zmat.block(0,0,row,1) = Z.centre;
    Zmat.block(0,1,row,gcol) = Z.generators;
    Eigen::MatrixXd quadMat, centre(row,1), gen(row,int((gcol*gcol + 3*gcol)*0.5));
    for(int i =0;i<H_mid.size();i++)
    {
        quadMat = Zmat.transpose() * H_mid[i] * Zmat; // (gcol+1) x (gcol+1)
        centre(i,0) = quadMat(0,0) + 0.5 * quadMat.diagonal().tail(gcol).sum();
        gen.row(i).head(gcol) = 0.5 * quadMat.diagonal().tail(gcol);
        int count =0;
        for(int j=0;j<gcol+1;j++)
        {
            gen.row(i).segment(gcol+count,gcol-j) = quadMat.row(j).segment(j+1,gcol-j)  + quadMat.col(j).segment(j+1,gcol-j).transpose();
            count += gcol - j;
        }
    }
    Zq.centre = centre;
    Zq.generators = gen;
    return mstom::deletezeros(Zq);
}

Eigen::VectorXd maxAbs(mstom::intervalMatrix IH){
    Eigen::MatrixXd Mtemp2(IH.lb.rows(),2);
    Mtemp2.col(0) = IH.lb;
    Mtemp2.col(1) = IH.ub;
    return Mtemp2.cwiseAbs().rowwise().maxCoeff();
}

extern int ZorDeltaZ;

Eigen::VectorXd compute_L_Hat1(mstom::zonotope Rtotal1, Eigen::VectorXd x_bar, int state_dim, Eigen::VectorXd uin){
    mstom::intervalMatrix RIH = mstom::IntervalHull(Rtotal1);
    Eigen::VectorXd L_hat(state_dim);
    {
        Eigen::VectorXd Gamma;
        if (ZorDeltaZ)  //1 if Z, 0 if deltaZ;
            Gamma = (Rtotal1.centre - x_bar).cwiseAbs() + Rtotal1.generators.cwiseAbs().rowwise().sum();
        else
            Gamma = (Rtotal1.centre).cwiseAbs() + Rtotal1.generators.cwiseAbs().rowwise().sum();
        
        Eigen::MatrixXd J_abs_max[state_dim]; 
        compute_J_abs_max(RIH, J_abs_max, uin); 
        for(int i=0; i<state_dim;i++)
        {
            L_hat(i) = 0.5 * Gamma.transpose() * J_abs_max[i] * Gamma;    
        }
    }
    return L_hat;
}

Eigen::VectorXd compute_L_Hat3(std::vector<vnodelp::interval> Kprime, std::vector<vnodelp::interval> uin){
    // global L_hat computation
     int dim = Kprime.size();
    Eigen::VectorXd L_hat(dim);
    {
        
        Eigen::VectorXd c(dim), lb(dim), ub(dim);
        for(int i=0;i<dim;i++)
        {
            lb(i) = vnodelp::inf(Kprime[i]);
            ub(i) = vnodelp::sup(Kprime[i]);
        }
        c = (lb + ub) * 0.5;
        mstom:: zonotope Z(c, ub-lb);
        mstom::intervalMatrix RIH = mstom::IntervalHull(Z);
        std::cout << "Z \n";
        Z.display();
        std::cout << "(RIH.ub-RIH.lb)*0.5\n "<< (RIH.ub-RIH.lb)*0.5 << endl;
        Eigen::VectorXd Gamma;
        Gamma = (0.5 * (ub-lb)) + Z.generators.cwiseAbs().rowwise().sum();
        Eigen::MatrixXd J_abs_max[dim]; //<-----------
        compute_J_abs_max(RIH, J_abs_max, uin);  //<----------
        for(int i=0; i<dim;i++)
        {
            L_hat(i) = 0.5 * Gamma.transpose() * J_abs_max[i] * Gamma;    
        }
        
    }
    return L_hat;
}

Eigen::VectorXd compute_L_Hat2(mstom::zonotope Rtotal1, Eigen::VectorXd x_bar, int state_dim, Eigen::VectorXd u){
    // 2nd L_hat computation method (less interval arithmatic)
    mstom::intervalMatrix RIH = mstom::IntervalHull(Rtotal1);
    Eigen::VectorXd L_hat(state_dim);
    {
        std::vector<vnodelp::iMatrix> H;
        compute_H(RIH, H, u);
        int dim = H[1].size();
        std::vector<Eigen::MatrixXd> H_mid, H_rad;
        Eigen::MatrixXd Mtemp(dim,dim);
        vnodelp::pMatrix pMtemp;
        vnodelp::sizeM(pMtemp,dim);
        vnodelp::pVector pV;
        vnodelp::sizeV(pV,dim);
        for(int i=0;i<H.size();i++)
        {
            vnodelp::midpoint(pMtemp, H[i]);
            H_mid.push_back(pMatrix_to_MatrixXd(pMtemp));
            for(int ii=0;ii<dim;ii++)
            {
                vnodelp::rad(pV, H[i][ii]);
                Mtemp.block(ii,0,1,dim) = pV_to_V(pV).transpose();
            }
            H_rad.push_back(Mtemp);
        }
        mstom::zonotope Zq = compute_quad(Rtotal1-x_bar, H_mid);
        mstom::zonotope error_mid;
        error_mid.centre = 0.5 * Zq.centre;
        error_mid.generators = 0.5 * Zq.generators;
        Eigen::VectorXd dz_abs = maxAbs(mstom::IntervalHull(Rtotal1-x_bar));   // edited 16Jan18
        Eigen::VectorXd error_rad(H.size());
        for(int i=0;i<H.size();i++)
            error_rad(i) = 0.5 * dz_abs.transpose() * H_rad[i] * dz_abs;
        mstom::zonotope error  = error_mid + mstom::zonotope(Eigen::VectorXd::Zero(dim), 2*error_rad);
        L_hat = maxAbs(mstom::IntervalHull(error));
    }
    
    return L_hat;
}

mstom::zonotope compute_Rerr_bar(int state_dim, mstom::intervalMatrix& Data_interm, mstom::zonotope& Rhomt, Eigen::VectorXd x_bar, Eigen::VectorXd f_bar, Eigen::VectorXd u, Eigen::VectorXd& L_hat, int LinErrorMethod, mstom::zonotope& F_tilde_f_bar){
    mstom::zonotope Rhom;
    Eigen::VectorXd appliedError = L_hat_previous;  // 06 March 18
    
    mstom::zonotope Verror, RerrAE, Rtotal1;
    Eigen::VectorXd trueError;
    Verror.centre = Eigen::VectorXd::Zero(state_dim);
    double perfIndCurr = 2;
    while(perfIndCurr > 1)
    {
        Rhom = Rhomt;
        if((f_bar-appliedError).maxCoeff() > 0 || (f_bar+appliedError).minCoeff() < 0)
            Rhom = Rhom + F_tilde_f_bar;    // when f_bar + [-L,L] does not contain origin
        Verror.generators = Eigen::MatrixXd(appliedError.asDiagonal());
        RerrAE = Data_interm * Verror;
        Rtotal1 = (RerrAE + Rhom);
         if(LinErrorMethod == 1)
            trueError = compute_L_Hat1(Rtotal1, x_bar, state_dim,u);
        else
            trueError = compute_L_Hat2(Rtotal1, x_bar, state_dim,u);
        perfIndCurr = (trueError.cwiseProduct(appliedError.cwiseInverse())).maxCoeff(); // max(trueError./appliedError)
        appliedError = 1.1 * trueError;   
    }
    L_hat = trueError;
    L_hat_previous = trueError;
    Verror.generators = Eigen::MatrixXd(trueError.asDiagonal());
    mstom::zonotope Rerror = Data_interm * Verror;
    return Rerror;
}

mstom::zonotope compute_L_hatB(int state_dim, Eigen::VectorXd& x_bar, mstom::zonotope& Z0, mstom::zonotope& exprAX0, double r, Eigen::VectorXd& fAx_bar,double Datab, double Datac, double Datad, int LinErrorMethod, Eigen::VectorXd& L_hat, Eigen::VectorXd u){
    // Guernic Girard
    Eigen::VectorXd appliedError = Eigen::MatrixXd::Constant(state_dim,1,0);
    mstom::zonotope AE; // zonotope(appliedError)
    AE.centre = Eigen::VectorXd::Zero(state_dim);
    Eigen::MatrixXd Mtemp(state_dim,2);
    mstom::zonotope B;  // unit ball in infinity norm: unit hyperrectangle
    B.centre = Eigen::VectorXd::Zero(state_dim);
    B.generators = Eigen::MatrixXd((Eigen::MatrixXd::Constant(state_dim,1,1)).asDiagonal());
    Eigen::VectorXd trueError;
    double Rv;
    mstom::zonotope V;
    double perfIndCurr = 2;
    while(perfIndCurr > 1)
    {
        AE.generators = Eigen::MatrixXd(appliedError.asDiagonal());
        V = AE + fAx_bar;
        Mtemp.col(0) = fAx_bar-appliedError;
        Mtemp.col(1) = fAx_bar+appliedError;
        Rv = Mtemp.cwiseAbs().maxCoeff();
        double alpha = Datac + Datad * Rv;
        mstom::zonotope omega0 = mstom::convexHull(Z0, (exprAX0+(V*r)+(B*alpha)));
        if(LinErrorMethod == 1)
            trueError = compute_L_Hat1(omega0, x_bar, state_dim,u);
        else
            trueError = compute_L_Hat2(omega0, x_bar, state_dim,u);
         perfIndCurr = (trueError.cwiseProduct(appliedError.cwiseInverse())).maxCoeff(); // max(trueError./appliedError)
        appliedError = 1.1 * trueError; 
     }
    L_hat = trueError;
    
    AE.generators = Eigen::MatrixXd(trueError.asDiagonal());
    V = AE + fAx_bar;
    Mtemp.col(0) = fAx_bar-trueError;
    Mtemp.col(1) = fAx_bar+trueError;
    Rv = Mtemp.cwiseAbs().maxCoeff();
    
    double beta = Datad * Rv;
    mstom::zonotope Rtau = exprAX0 + (V*r) + (B*beta);
    return Rtau;
}

//----------------------------------------------------------------------------------------
//########################################################################################
// Reachable set


std::vector<mstom::zonotope> PlotStorage;
int ZorDeltaZ;  // 1 if Z, 0 if deltaZ
extern double LHatTimeavg;    //defined in Abstraction.hh
extern int countavg;                    //defined in Abstraction.hh, count for average of time

void splitz(mstom::zonotope Z0, mstom::zonotope& Z01, mstom::zonotope& Z02, Eigen::MatrixXf::Index maxIndex)
{
    Z01.centre = Z0.centre - 0.5 * Z0.generators.col(maxIndex);
    Z02.centre = Z0.centre + 0.5 * Z0.generators.col(maxIndex);
    Z01.generators = Z0.generators;
    Z02.generators = Z0.generators;
    Z01.generators.col(maxIndex) = 0.5 * Z0.generators.col(maxIndex);
    Z02.generators.col(maxIndex) = 0.5 * Z0.generators.col(maxIndex);
}



template<class state_type>
Eigen::VectorXd computeM(double tau, state_type lower_left, state_type upper_right, state_type inp_lower_left, state_type inp_upper_right){
    int state_dim = 2;
    int input_dim = 1;
    Eigen::VectorXd M(state_dim), Mb(state_dim);
    std::vector<vnodelp::interval> xx(state_dim), x(state_dim), u(input_dim), Kprime(state_dim), Kbprime(state_dim);
    for(int i=0;i<state_dim;i++)
    {
        x[i] = vnodelp::interval(lower_left[i], upper_right[i]);
    }
    for(int i=0; i<input_dim;i++)
        u[i] = vnodelp::interval(inp_lower_left[i], inp_upper_right[i]);
    xx = funcLj_system(x, u, xx);
    for(int i=0;i<state_dim;i++)
    {
        M(i) = vnodelp::mag(xx[i]);
        Kprime[i] = vnodelp::interval(lower_left[i]-M(i)*tau, upper_right[i]+M(i)*tau);
    }
    x = Kprime;
    std::cout << "M\n"<< M << endl;
    int checkval = 0;
    while(checkval < state_dim)
    {
        checkval = 0;   //0 if further iteration needed else >0
         xx = funcLj_system(x, u, xx);
        for(int i=0;i<state_dim;i++)
        {
            Mb(i) = vnodelp::mag(xx[i]);
            Kbprime[i] = vnodelp::interval(lower_left[i]-Mb(i)*tau, upper_right[i]+Mb(i)*tau);
             std::cout<< "bool: " << (Mb(i) <= M(i)) << endl;
            std::cout << "(Mb(i)-M(i)) = " << (Mb(i) - M(i))<< endl;
            if(Mb(i) <= M(i))
            {
                  checkval++;
            }
        }
        M = Mb; // Mb will never be less than M
        std::cout << "Mb\n"<< Mb <<"\ncheckval= "<< checkval << endl ;
        x=Kbprime;
    }
    Eigen::VectorXd gL_hat = compute_L_Hat3(Kbprime, u);
    return gL_hat;
}


int computeKprime(double tau, vnodelp::interval* xin,  vnodelp::interval u[], int dim, vnodelp::interval* Kbprime, int KprimeLimit, vnodelp::interval* x_initial){
    double M[dim], Mb[dim];
    vnodelp::interval xx[dim], Kprime[dim], *x;
    x = xin;
    funcLj_system(x, u, xx);
    for(int i=0;i<dim;i++)
    {
        M[i] = vnodelp::mag(xx[i]);
        Kprime[i] = xin[i] + vnodelp::interval(-M[i]*tau, M[i]*tau);
    }
    x = Kprime;
     
    int checkval = 0;
    int countchk = 0;
    while(checkval < dim)
    {
        countchk++;
        checkval = 0;   //0 if further iteration needed else >0
        funcLj_system(x, u, xx);
        for(int i=0;i<dim;i++)
        {
            Mb[i] = vnodelp::mag(xx[i]);
            Kbprime[i] = xin[i] + vnodelp::interval(-Mb[i]*tau, Mb[i]*tau);
            
            if(std::abs(vnodelp::sup(Kbprime[i])) > (KprimeLimit*std::abs(vnodelp::sup(x_initial[i]))))
            {
                 return 1;
            }
              if(Mb[i] - M[i] <= pow(10,-4))
            {
                checkval++;
            }
        }
        // Mb will never be less than M
        for(int i=0;i<dim;i++)
        {
            M[i] = Mb[i];
        }
         x = Kbprime;
    }
     return 0;
}

template<class Lugbx, class Lugbu>
void computeLu(Lugbx xin, Lugbu uin, Lugbx rin, double tau_in, int dim, int dimInput, double* Lu){
    // if M keep rising, then split tau
    vnodelp::interval x[dim], u[dimInput], Kprime[dim], *xp;
    int KprimeLimit = std::pow(10,4);
    for(int i=0;i<dim;i++)
    {
        x[i] = vnodelp::interval(xin[i]-rin[i],xin[i]+rin[i]);  // m_z already included in rin
    }
    for(int i=0;i<dimInput;i++)
        u[i] = vnodelp::interval(uin[i]);
    double tau = tau_in;
    int checkval = 0;
    int chk = 1;
    while(chk == 1)
    {
        xp = x;
        checkval = 0;
        int tauDivision = tau_in/tau;
        for(int i = 0;i<tauDivision;i++)
        {          
            chk = computeKprime(tau,xp,u,dim,Kprime,KprimeLimit, x);
            xp = Kprime;
            if(chk == 1)
                break;
        }
        tau = tau/2;
    }
    computeJacobian_Lu_array2(Kprime, u, dim, 0, Lu); // returns data in Lu
}

template<class Lugbx, class Lugbu, class F4>
Eigen::MatrixXd LuOverSS(Lugbx& lower_left, Lugbx& upper_right, Lugbu& uin, int dim, int dimInput){
    //    std::vector<vnodelp::interval> x(dim), u(dimInput);
    vnodelp::interval x[dim], u[dimInput];
    for(int i=0;i<dim;i++)
        x[i] = vnodelp::interval(lower_left[i], upper_right[i]);
    for(int i=0;i<dimInput;i++)
        u[i] = vnodelp::interval(uin[i]);
    Eigen::MatrixXd L(dim,dim);
    computeJacobian_Lu_array(x,u,L,dim);
    return L;
}

template<class Lugbx, class Lugbu>
Eigen::MatrixXd LuOverSS_array(Lugbx& lower_left, Lugbx& upper_right, Lugbu& uin, int& dim, int& dimInput){
    // using arrays
    vnodelp::interval x[dim], u[dimInput];
    for(int i=0;i<dim;i++)
        x[i] = vnodelp::interval(lower_left[i], upper_right[i]);
    for(int i=0;i<dimInput;i++)
        u[i] = uin[i];
    Eigen::MatrixXd L(dim,dim);
    computeJacobian_Lu_array(x,u,L, dim);
    return L;
}

template<class Lugbx, class Lugbu>
void LuOverSS_array2(Lugbx& lower_left, Lugbx& upper_right, Lugbu& uin, int& dim, int& dimInput, int jin, double* LuStore){
    // without eigen::matrix
    // using arrays
    vnodelp::interval x[dim], u[dimInput];
    for(int i=0;i<dim;i++)
        x[i] = vnodelp::interval(lower_left[i], upper_right[i]);
    for(int i=0;i<dimInput;i++)
        u[i] = uin[i];
    computeJacobian_Lu_array2(x,u, dim, jin, LuStore);
}


template<class CL>
mstom::zonotope one_iteration(mstom::zonotope Z0, Eigen::VectorXd u, int state_dim, double r, int& p, const Eigen::VectorXd L_bar, std::vector<mstom::zonotope>& stora, int& count1, int LinErrorMethod, CL& L_hat_storage, const Eigen::VectorXd& ss_eta)
{
    TicToc timehat;
    count1++;
    Eigen::VectorXd c = Z0.centre;
    std::vector<double> cv ;  // to pass c as c++ vector to func
    mstom::VXdToV(cv, c);
    Eigen::VectorXd x_bar(state_dim);
    x_bar = c + 0.5 * r * funcLj_system(c, u, x_bar);
    
    Eigen::MatrixXd A(state_dim,state_dim);
    
    double A_array[state_dim*state_dim];
    
    computeJacobian(A_array, x_bar, u);
    for(int i=0;i<state_dim;i++)
        for(int j=0;j<state_dim;j++)
            A(i,j) = A_array[i*state_dim+j];
    Eigen::VectorXd f_bar(state_dim);
    f_bar = funcLj_system(x_bar, u, f_bar);
    mstom::zonotope Rtotal_tp;
    Eigen::VectorXd L_hat;
    
#if 1   // 1 if exponential as series truncation; 0 if exponential matrix of eigen (GuernicGirard) 
    mstom::intervalMatrix Er;   //E(r)
    mstom::intervalMatrix Data_interm;  //(Aˆ-1)*(exp(Ar) - I)
    double epsilone = mstom::compute_epsilon(A,r,p);
    int isOriginContained = 0;
    //    Ar_powers_fac; // pow(A*r,i)/factorial(i)
    
    double bound = mstom::p_adjust_Er_bound(A,r,p,epsilone);
    double Ar_powers_fac_arr[state_dim*state_dim*p];
    
    mstom::intervalMatrix expAr = mstom::matrix_exponential(A, r, p, epsilone, Er, Ar_powers_fac_arr, bound);  //exp(A*r);
    mstom::intervalMatrix F = mstom::compute_F(p, r, A, Er, Ar_powers_fac_arr);
    mstom::intervalMatrix F_tilde = mstom::compute_F_tilde(p,r,A,Er,isOriginContained, Ar_powers_fac_arr);
    mstom::zonotope F_tilde_f_bar = F_tilde * mstom::zonotope(f_bar, Eigen::VectorXd::Zero(state_dim));
    
    // Data_interm = (Aˆ-1)*(exp(Ar) - I)
    Data_interm = compute_Data_interm(Er, r, p, A, Ar_powers_fac_arr);
    mstom::zonotope Z0delta = Z0 + mstom::zonotope(-x_bar,Eigen::VectorXd::Zero(A.rows()));
    mstom::zonotope Rtrans = Data_interm * mstom::zonotope(f_bar, Eigen::VectorXd::Zero(state_dim));
    mstom::zonotope Rhom_tp = expAr * Z0delta + Rtrans ;
    mstom::zonotope Rtotal;
    
    mstom::zonotope Rhom = mstom::convexHull(Z0delta, Rhom_tp) + F * Z0delta + x_bar;   // without F_tilde
    ZorDeltaZ = 1;  //1 if Z, 0 if deltaZ
    mstom::zonotope RV;
    TicToc timeLhat;
    timeLhat.tic();
    RV = compute_Rerr_bar(state_dim, Data_interm, Rhom, x_bar, f_bar, u, L_hat,LinErrorMethod, F_tilde_f_bar);  // Rerr_bar; L_hat updated in the call
    LHatTimeavg += timeLhat.tocMST();
    Rtotal_tp = Rhom_tp + RV  + x_bar;
    
#else
    ZorDeltaZ = 0;    //1 if Z, 0 if deltaZ
    double RdeltaX0 = ((c-x_bar).cwiseAbs() + (ss_eta/2)).maxCoeff();
    double AinfNorm = A.cwiseAbs().rowwise().sum().maxCoeff();
    double Datab = std::exp(r * AinfNorm)-1-r*AinfNorm;
    double Datac = Datab * RdeltaX0;
    double Datad = Datab / AinfNorm;
    mstom::zonotope Z0delta = Z0-x_bar;
    mstom::zonotope exprAdeltaX0 = (r*A).exp() * (Z0delta)  ; // exp(r*A) * Z0 ;
    Rtotal_tp = compute_L_hatB(state_dim, x_bar, Z0delta, exprAdeltaX0, r, f_bar, Datab, Datac, Datad, LinErrorMethod, L_hat, u) + x_bar;
#endif
    
    if(L_hat_storage.size() < 1)
        L_hat_storage.push_back(L_hat(1));  
    
    Eigen::VectorXd Ltemp = L_bar - L_hat;
    bool aa = true;
    Eigen::MatrixXf::Index maxIndex;
    for(int i=0;i<state_dim;i++)
    {
        if (Ltemp(i) < 0)
        {
            aa = false;
            L_hat.maxCoeff(&maxIndex);
            break;
        }
        else
        {
            aa = true;
        }
    }
    if(aa==true)
    {
        stora.push_back(Rtotal_tp);
    }
    if(count1 > 4)  //<--------------------------
    {
        std::cout<< "Count reached limit"<< endl;
        return Rtotal_tp;
    }
    mstom::zonotope Z01, Z02;
    if(aa == false)
    {
        splitz(Z0,Z01,Z02,maxIndex);
        std::cout << "\n#########\n##  Split  ##\n###########\n\n";
        one_iteration(Z01, u, state_dim, r, p, L_bar, stora, count1, LinErrorMethod, L_hat_storage, ss_eta);
        one_iteration(Z02, u, state_dim, r, p, L_bar, stora, count1, LinErrorMethod, L_hat_storage, ss_eta);
    }
    
    return Rtotal_tp;
}


template<class state_type, class input_type, class CL>
mstom::zonotope ReachableSet(int dim, int dimInput, const double tau, state_type& rr, state_type& x, input_type uu, CL& L_hat_storage)
{
    int LinErrorMethod = 1; // 1 or 2   <--------------
    int p = 5; // 3; matrix exponential terminated after p terms
    // appliedError = 1.1 * trueError for both Rerr_bar and L_hatB
    // appliedError begins with _0_ for Rerr_bar, with 0 for L_hatB
    // p increased till E(r) < pow(10,-5)  
    double l_bar = 1000;   //max above which splits
   
    int state_dim = dim;
    int input_dim = dimInput;
    double finaltime = tau;
    double r = tau;    //sampling period = r
    
    Eigen::VectorXd L_bar(state_dim);
    L_bar = Eigen::VectorXd::Constant(state_dim,1,l_bar);
    Eigen::VectorXd c(state_dim);  // c = centre of the cell
    Eigen::VectorXd ss_eta(state_dim); // state space eta
    Eigen::VectorXd u(input_dim);  //the selected value of input
    //c << 3,0;
    for(int i=0;i<state_dim;i++)
    {
        c(i) = x[i];
        ss_eta(i) = 2 * rr[i];
    }
    for(int i=0;i<input_dim;i++)
        u(i) = uu[i];
     
    mstom::zonotope Z0(c,ss_eta);
    std::vector<mstom::zonotope> stora;  // storage for Reachable sets (at r)
    
    mstom::zonotope Zn;
    for(int i=0;i<int(finaltime/r);i++)
    {
        int count1 = 0;
        Zn = one_iteration(Z0,u,state_dim,r,p,L_bar,stora,count1, LinErrorMethod, L_hat_storage, ss_eta);
        Z0 = Zn;
    }
    
    // forming convex hull of all zonotopes in stora
    mstom::zonotope Zf = mstom::convexHull(stora);
    mstom::intervalMatrix IHM = IntervalHull(Zf);
    for(int i=0;i<state_dim;i++)
    {
        rr[i] = (IHM.ub(i) - IHM.lb(i)) * 0.5;
        x[i] = IHM.lb(i) + rr[i];
    }
    return Zn;
}


#endif /* ReachableSet_h */
