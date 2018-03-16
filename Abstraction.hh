
// mst: edited 28 Dec 17
// compute_gb2() for zonotope based abstraction

/*
 * Abstraction.hh
 *
 *  created: Jan 2017
 *   author: Matthias Rungger
 */

/** @file **/
#ifndef ABSTRACTION_HH_
#define ABSTRACTION_HH_

#include <iostream>
#include <cstring>
#include <memory>
#include <vector>

#include "UniformGrid.hh"
#include "TransitionFunction.hh"

#include "ReachableSet.h"
//#include "LuGrowthbound.h"
#include <forward_list>
#include "TicToc.hh"
#include "RungeKutta4.hh"

double ttimeavg = 0;    // total time average in reachable set computation
double LHatTimeavg = 0;     // L_hat time average
int countavg = 0;   // count for average of time
Eigen::VectorXd L_hat_previous;

/** @namespace scots **/ 
namespace scots {

/** @cond **/
/* default parameter for the third parameter of Abstraction::compute_gb  */
namespace params {
  auto avoid_abs = [](const abs_type&) noexcept {return false;};
}
/** @endcond **/


/**
 * @class Abstraction
 * 
 * @brief Computation of the transition function of symbolic model/abstraction
 *
 * It implements in compute_gb the computation of the transition function based
 * on a growth bound. Additionally, it provides the functions print_post and
 * print_post_gb to print the center of the cells that cover the attainable set
 * which is computed from the growth bound.
 *
 * See 
 * - the manual in <a href="./../../manual/manual.pdf">manual.pdf</a>
 * - http://arxiv.org/abs/1503.03715 for theoretical background 
 *
 **/
template<class state_type, class input_type>
class Abstraction {
private:
  /* grid information of state alphabet */
  const UniformGrid m_state_alphabet;
  /* grid information of input alphabet */
  const UniformGrid m_input_alphabet;
  /* measurement error bound */
  std::unique_ptr<double[]> m_z;
  /* print progress to the console (default m_verbose=true) */
  bool m_verbose=true;
  /* to display the progress of the computation of the abstraction */
  void progress(const abs_type& i, const abs_type& N, abs_type& counter) {
    if(!m_verbose)
      return;
    if(((double)i/(double)N*100)>=counter){
      if((counter%10)==0)
        std::cout << counter;
      else if((counter%2)==0) 
        std::cout << ".";
      counter++;
    }
    std::flush(std::cout); 
    if(i==(N-1))
      std::cout << "100\n";
  }
public:
  /* @cond  EXCLUDE from doxygen*/
  /* deactivate standard constructor */
  Abstraction() = delete;
  /* cannot be copied or moved */
  Abstraction(Abstraction&&) = delete;
  Abstraction(const Abstraction&) = delete;
  Abstraction& operator=(Abstraction&&)=delete;
  Abstraction& operator=(const Abstraction&)=delete;
  /* @endcond */

  /** @brief constructor with the abstract state alphabet and abstract input
   * alphabet (measurement error bound is set to zero)
   *
   *  @param state_alphabet - UniformGrid representing the abstract state alphabet
   *  @param input_alphabet - UniformGrid representing the abstract input alphabet
   **/
  Abstraction(const UniformGrid& state_alphabet,
              const UniformGrid& input_alphabet) :
              m_state_alphabet(state_alphabet),
              m_input_alphabet(input_alphabet),
              m_z(new double[state_alphabet.get_dim()]()) {
  /* default value of the measurement error 
     * (heurisitc to prevent rounding errors)*/
    for(int i=0; i<m_state_alphabet.get_dim(); i++)
      m_z[i]=m_state_alphabet.get_eta()[i]/1e10;
  }

  /** 
   * @brief computes the transition function
   *
   * @param[out] transition_function - the result of the computation
   *
   * @param[in] system_post - lambda expression of the form
   *                          \verbatim [] (state_type &x, const input_type &u) ->  void  \endverbatim
   *                          system_post(x,u) provides a numerical approximation of ODE 
   *                          solution at time tau with initial state x and input u \n
   *                          the result is stored in x
   *
   * @param[in] radius_post - lambda expression of the form
   *                          \verbatim [] (state_type &r, const state_type& x, const input_type &u) -> void  \endverbatim
   *                          radius_post(x,u) provides a numerical approximation of
   *                          the growth bound for the cell (with center x, radius  r) and input u\n
   *                          the result is stored in r
   *
   * @param[in] avoid  - OPTIONALLY provide lambda expression of the form
   *                     \verbatim [] (const abs_type &i) -> bool \endverbatim
   *                     returns true if the abstract state i is in the avoid
   *                     set; otherwise returns false
   *
   * The computation proceeds in two loops. In the first loop the cornerIDs are
   * comuted, which represent the cell IDs that cover the over-approximation of the attainable set:
   * \verbatim corner_IDs[i*M+j+0] \endverbatim = lower-left cell ID of the
   * integer hyper-interval of cell IDs that cover the over-approximation of the
   * attainable set of cell with ID=i under input ID=j
   * \verbatim corner_IDs[i*M+j+1] \endverbatim = upper-right  of the
   * integer hyper-interval of cell IDs that cover the over-approximation of the
   * attainable set of cell with ID=i under input ID=j
   * 
   * In the second loop the data members of the TransitionFunction are computed.
   * 
   **/
    
    template<class F1, class F2, class F3=decltype(params::avoid_abs)>
    void compute_gb(TransitionFunction& transition_function,
                    F1& system_post,
                    F2& radius_post,
                    F3& avoid=params::avoid_abs) {
        /* number of cells */
        abs_type N=m_state_alphabet.size();
        /* number of inputs */
        abs_type M=m_input_alphabet.size();
        /* number of transitions (to be computed) */
        abs_ptr_type T=0;
        /* state space dimension */
        int dim=m_state_alphabet.get_dim();
        /* for display purpose */
        abs_type counter=0;
        /* some grid information */
        std::vector<abs_type> NN=m_state_alphabet.get_nn();
        /* variables for managing the post */
        std::vector<abs_type> lb(dim);  /* lower-left corner */
        std::vector<abs_type> ub(dim);  /* upper-right corner */
        std::vector<abs_type> no(dim);  /* number of cells per dim */
        std::vector<abs_type> cc(dim);  /* coordinate of current cell in the post */
        /* radius of hyper interval containing the attainable set */
        state_type eta;
        state_type r;
        /* state and input variables */
        state_type x;
        input_type u;
        /* for out of bounds check */
        state_type lower_left;
        state_type upper_right;
        /* copy data from m_state_alphabet */
        for(int i=0; i<dim; i++) {
            eta[i]=m_state_alphabet.get_eta()[i];
            lower_left[i]=m_state_alphabet.get_lower_left()[i];
            upper_right[i]=m_state_alphabet.get_upper_right()[i];
        }
        /* init in transition_function the members no_pre, no_post, pre_ptr */
        transition_function.init_infrastructure(N,M);
        /* lower-left & upper-right corners of hyper rectangle of cells that cover attainable set */
        std::unique_ptr<abs_type[]> corner_IDs(new abs_type[N*M*2]());
        /* is post of (i,j) out of domain ? */
        std::unique_ptr<bool[]> out_of_domain(new bool[N*M]());
        /*
         * first loop: compute corner_IDs:
         * corner_IDs[i*M+j][0] = lower-left cell index of over-approximation of attainable set
         * corner_IDs[i*M+j][1] = upper-right cell index of over-approximation of attainable set
         */
        /* loop over all cells */
        for(abs_type i=0; i<N; i++) {
            /* is i an element of the avoid symbols ? */
            if(avoid(i)) {
                for(abs_type j=0; j<M; j++) {
                    out_of_domain[i*M+j]=true;
                }
                continue;
            }
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) {
                out_of_domain[i*M+j]=false;
                /* get center x of cell */
                m_state_alphabet.itox(i,x);
                /* cell radius (including measurement errors) */
                for(int k=0; k<dim; k++)
                    r[k]=eta[k]/2.0+m_z[k];
                /* current input */
                m_input_alphabet.itox(j,u);
                /* integrate system and radius growth bound */
                /* the result is stored in x and r */
                radius_post(r,x,u);
                
//                std::cout<< "radius post = ";
//                for(int ii = 0;ii<dim;ii++)
//                    std::cout<< r[ii] << ", ";
//                std::cout<< std::endl;
                
                system_post(x,u);   // <---------- mst
                /* determine the cells which intersect with the attainable set:
                 * discrete hyper interval of cell indices
                 * [lb[0]; ub[0]] x .... x [lb[dim-1]; ub[dim-1]]
                 * covers attainable set
                 */
                abs_type npost=1;
                for(int k=0; k<dim; k++) {
                    /* check for out of bounds */
                    double left = x[k]-r[k]-m_z[k];
                    double right = x[k]+r[k]+m_z[k];
                    if(left <= lower_left[k]-eta[k]/2.0  || right >= upper_right[k]+eta[k]/2.0)  {
                        out_of_domain[i*M+j]=true;
                        break;
                    }
                    
                    /* integer coordinate of lower left corner of post */
                    lb[k] = static_cast<abs_type>((left-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* integer coordinate of upper right corner of post */
                    ub[k] = static_cast<abs_type>((right-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* number of grid points in the post in each dimension */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total number of post */
                    npost*=no[k];
                    cc[k]=0;
                }
                corner_IDs[i*(2*M)+2*j]=0;
                corner_IDs[i*(2*M)+2*j+1]=0;
                if(out_of_domain[i*M+j])
                    continue;
                
                /* compute indices of post */
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    /* (i,j,q) is a transition */
                    /* increment number of pres for (q,j) */
                    transition_function.m_no_pre[q*M+j]++;
                    /* store id's of lower-left and upper-right cell */
                    if(k==0)
                        corner_IDs[i*(2*M)+2*j]=q;
                    if(k==npost-1)
                        corner_IDs[i*(2*M)+2*j+1]=q;
                }
                /* increment number of transitions by number of post */
                T+=npost;
                transition_function.m_no_post[i*M+j]=npost;
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "1st loop: ";
            }
            progress(i,N,counter);
        }
        /* compute pre_ptr */
        abs_ptr_type sum=0;
        for(abs_type i=0; i<N; i++) {
            for(abs_type j=0; j<M; j++) {
                sum+=transition_function.m_no_pre[i*M+j];
                transition_function.m_pre_ptr[i*M+j]=sum;
            }
        }
        /* allocate memory for pre list */
        transition_function.init_transitions(T);
        
        /* second loop: fill pre array */
        counter=0;
        for(abs_type i=0; i<N; i++) {
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) {
                /* is x an element of the overflow symbols ? */
                if(out_of_domain[i*M+j])
                    continue;
                /* extract lower-left and upper-bound points */
                abs_type k_lb=corner_IDs[i*2*M+2*j];
                abs_type k_ub=corner_IDs[i*2*M+2*j+1];
                abs_type npost=1;
                
                /* cell idx to coordinates */
                for(int k=dim-1; k>=0; k--) {
                    /* integer coordinate of lower left corner */
                    lb[k]=k_lb/NN[k];
                    k_lb=k_lb-lb[k]*NN[k];
                    /* integer coordinate of upper right corner */
                    ub[k]=k_ub/NN[k];
                    k_ub=k_ub-ub[k]*NN[k];
                    /* number of grid points in each dimension in the post */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total no of post of (i,j) */
                    npost*=no[k];
                    cc[k]=0;
                    
                }
                
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    /* (i,j,q) is a transition */
                    transition_function.m_pre[--transition_function.m_pre_ptr[q*M+j]]=i;
                }
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "2nd loop: ";
            }
            progress(i,N,counter);
        }
    }

    
    
  template<class F1, class F3=decltype(params::avoid_abs)>
  void compute_gbLu(TransitionFunction& transition_function,
                  F1& system_post, 
                  double tau,   // mst 31 Jan18
                  F3& avoid=params::avoid_abs) {
    /* number of cells */
    abs_type N=m_state_alphabet.size(); 
    /* number of inputs */
    abs_type M=m_input_alphabet.size();
    /* number of transitions (to be computed) */
    abs_ptr_type T=0; 
    /* state space dimension */
    int dim=m_state_alphabet.get_dim();
      int dimInput = m_input_alphabet.get_dim();    //mst 31 Jan18;L(u) growth bound
      
    /* for display purpose */
    abs_type counter=0;
    /* some grid information */
    std::vector<abs_type> NN=m_state_alphabet.get_nn();
    /* variables for managing the post */
    std::vector<abs_type> lb(dim);  /* lower-left corner */
    std::vector<abs_type> ub(dim);  /* upper-right corner */
    std::vector<abs_type> no(dim);  /* number of cells per dim */
    std::vector<abs_type> cc(dim);  /* coordinate of current cell in the post */
    /* radius of hyper interval containing the attainable set */
    state_type eta;
    state_type r;
    /* state and input variables */
    state_type x;
    input_type u;
    /* for out of bounds check */
    state_type lower_left;
    state_type upper_right;
    /* copy data from m_state_alphabet */
    for(int i=0; i<dim; i++) {
      eta[i]=m_state_alphabet.get_eta()[i];
      lower_left[i]=m_state_alphabet.get_lower_left()[i];
      upper_right[i]=m_state_alphabet.get_upper_right()[i];
    }
    /* init in transition_function the members no_pre, no_post, pre_ptr */ 
    transition_function.init_infrastructure(N,M);
    /* lower-left & upper-right corners of hyper rectangle of cells that cover attainable set */
    std::unique_ptr<abs_type[]> corner_IDs(new abs_type[N*M*2]());  // <--- ?mst why multiplication by 2
    /* is post of (i,j) out of domain ? */
    std::unique_ptr<bool[]> out_of_domain(new bool[N*M]());
      
      
      /* computation of the growth bound (the result is stored in r)  */
      auto radius_post2 = [&tau, &dim](state_type &r, const state_type&, const input_type &u, Eigen::MatrixXd &Lu) noexcept {
          
          auto rhs = [&Lu, &tau, &dim](state_type &drdt,  const state_type &r, const input_type &u) noexcept {
              for(int i=0;i<dim;i++)
              {
                  drdt[i] = 0;
                  for(int j=0;j<dim;j++)
                      drdt[i] = drdt[i] + Lu(i,j)*r[j];
                  drdt[i] = drdt[i];
              }
          };
          scots::runge_kutta_fixed4(rhs,r,u,dim,tau);
      };
      
      auto radius_post2b = [&tau, &dim](state_type &r, const state_type&, const input_type &u, double* Lu) noexcept {
          
          auto rhs = [&Lu, &tau, &dim](state_type &drdt,  const state_type &r, const input_type &u) noexcept {
              for(int i=0;i<dim;i++)
              {
                  drdt[i] = 0;
                  for(int j=0;j<dim;j++)
                      drdt[i] = drdt[i] + Lu[i*dim+j]*r[j];
                  drdt[i] = drdt[i];
              }
          };
          scots::runge_kutta_fixed4(rhs,r,u,dim,tau);
      };


      
//      std::vector<Eigen::MatrixXd> LuStore(M); // store L(u) as the number of inputs
//      Eigen::MatrixXd* LuStore = new Eigen::MatrixXd [M];
      double* LuStore = new double[M*dim*dim];
      
      
    /*
     * first loop: compute corner_IDs:
     * corner_IDs[i*M+j][0] = lower-left cell index of over-approximation of attainable set 
     * corner_IDs[i*M+j][1] = upper-right cell index of over-approximation of attainable set 
     */
    /* loop over all cells */
    for(abs_type i=0; i<N; i++) {
      /* is i an element of the avoid symbols ? */
      if(avoid(i)) {
        for(abs_type j=0; j<M; j++) {
          out_of_domain[i*M+j]=true;
        }
        continue;
      }
      /* loop over all inputs */
      for(abs_type j=0; j<M; j++) {
//          std::cout << endl << "j = " << j << endl;
        out_of_domain[i*M+j]=false;
        /* get center x of cell */
        m_state_alphabet.itox(i,x);
        /* cell radius (including measurement errors) */
        for(int k=0; k<dim; k++)
          r[k]=eta[k]/2.0+m_z[k];
        /* current input */
        m_input_alphabet.itox(j,u);
        /* integrate system and radius growth bound */
        /* the result is stored in x and r */
          
          countavg++;       // mst 25 Jan18
          TicToc timet2;    // mst 25 Jan18
          timet2.tic();     // mst 25 Jan18
          
//        radius_post(r,x,u);
          
//          std::cout<< "r = ";
//          for(int ii = 0;ii<dim;ii++)
//              std::cout<< r[ii] << ", ";
//           std::cout<< std::endl;
//          std::cout<< "x = ";
//          for(int ii = 0;ii<dim;ii++)
//              std::cout<< x[ii] << ", ";
//          std::cout<< std::endl;
//          std::cout<< "u = ";
//          for(int ii = 0;ii<dimInput;ii++)
//              std::cout<< u[ii] << ", ";
//          std::cout<< std::endl;
          
//          double Lu[dim*dim];
//          computeLuMRerr(x,u,tau,r, Lu); // Using vnodelp
          
//          Eigen::MatrixXd Lu = computeLu(x,u,r,tau,funcExpre,dim, dimInput);    //Mtau

          double Lu[dim*dim];
          computeLu(x,u,r,tau,dim, dimInput, Lu);    //Mtau using array
          
//          Eigen::MatrixXd Lu = LuOverSS(lower_left, upper_right, u, funcExpre, dim, dimInput);  // without computation of K'
          
//          // storing L(u)
//          Eigen::MatrixXd Lu;
//          if(i==0){
////              LuStore[j] = LuOverSS(lower_left, upper_right, u, funcExpre, dim, dimInput);  // without K'
//              LuStore[j] = LuOverSS_array(lower_left, upper_right, u, dim, dimInput);
//          }
//          Lu = LuStore[j];
          
//          // storing L(u); without using eigen matrix, instead array
//          double* Lu;
//          if(i==0){
//              //              LuStore[j] = LuOverSS(lower_left, upper_right, u, dim, dimInput);  // without K'
//              LuOverSS_array2(lower_left, upper_right, u, dim, dimInput, j, LuStore);
//          }
//          Lu = &LuStore[j*dim*dim];
          
          radius_post2b(r,x,u,Lu);  // Lu as double*
//          radius_post2(r,x,u,Lu);
          
         
//          std::cout<< "radius post = ";
//          for(int ii = 0;ii<dim;ii++)
//              std::cout<< r[ii] << ", ";
//          std::cout<< std::endl;
          
//          compute_radius_post(r,x,u,tau, dim, dimInput, funcExpre )   // 31 Jan18; for L(u); growth bound
          system_post(x,u);
          
          ttimeavg += timet2.tocMST();  // mst 25 Jan18
          
          
          
        /* determine the cells which intersect with the attainable set: 
         * discrete hyper interval of cell indices 
         * [lb[0]; ub[0]] x .... x [lb[dim-1]; ub[dim-1]]
         * covers attainable set 
         */
        abs_type npost=1;
        for(int k=0; k<dim; k++) {
          /* check for out of bounds */
          double left = x[k]-r[k]-m_z[k];
          double right = x[k]+r[k]+m_z[k];
          if(left <= lower_left[k]-eta[k]/2.0  || right >= upper_right[k]+eta[k]/2.0)  {
            out_of_domain[i*M+j]=true;
            break;
          } 

          /* integer coordinate of lower left corner of post */
          lb[k] = static_cast<abs_type>((left-lower_left[k]+eta[k]/2.0)/eta[k]);
          /* integer coordinate of upper right corner of post */
          ub[k] = static_cast<abs_type>((right-lower_left[k]+eta[k]/2.0)/eta[k]);
          /* number of grid points in the post in each dimension */
          no[k]=(ub[k]-lb[k]+1);
          /* total number of post */
          npost*=no[k];
          cc[k]=0;
        }
        corner_IDs[i*(2*M)+2*j]=0;
        corner_IDs[i*(2*M)+2*j+1]=0;
        if(out_of_domain[i*M+j]) 
          continue;

        /* compute indices of post */
        for(abs_type k=0; k<npost; k++) {
          abs_type q=0;
          for(int l=0; l<dim; l++) 
            q+=(lb[l]+cc[l])*NN[l];
          cc[0]++;
          for(int l=0; l<dim-1; l++) {
            if(cc[l]==no[l]) {
              cc[l]=0;
              cc[l+1]++;
            }
          }
          /* (i,j,q) is a transition */    
          /* increment number of pres for (q,j) */ 
          transition_function.m_no_pre[q*M+j]++;
          /* store id's of lower-left and upper-right cell */
          if(k==0)
            corner_IDs[i*(2*M)+2*j]=q;
          if(k==npost-1)
            corner_IDs[i*(2*M)+2*j+1]=q;
        }
        /* increment number of transitions by number of post */
        T+=npost;
        transition_function.m_no_post[i*M+j]=npost;
      }
      /* print progress */
      if(m_verbose) {
        if(counter==0)
          std::cout << "1st loop: ";
      }
      progress(i,N,counter);
    }
      delete[] LuStore;
      
    /* compute pre_ptr */
    abs_ptr_type sum=0;
    for(abs_type i=0; i<N; i++) {
      for(abs_type j=0; j<M; j++) {
        sum+=transition_function.m_no_pre[i*M+j];
        transition_function.m_pre_ptr[i*M+j]=sum;
      }
    }
    /* allocate memory for pre list */
    transition_function.init_transitions(T);

    /* second loop: fill pre array */
    counter=0;
    for(abs_type i=0; i<N; i++) {
      /* loop over all inputs */
      for(abs_type j=0; j<M; j++) {
      /* is x an element of the overflow symbols ? */
        if(out_of_domain[i*M+j]) 
          continue;
        /* extract lower-left and upper-bound points */
        abs_type k_lb=corner_IDs[i*2*M+2*j];
        abs_type k_ub=corner_IDs[i*2*M+2*j+1];
        abs_type npost=1;

        /* cell idx to coordinates */
        for(int k=dim-1; k>=0; k--) {
          /* integer coordinate of lower left corner */
          lb[k]=k_lb/NN[k];
          k_lb=k_lb-lb[k]*NN[k];
          /* integer coordinate of upper right corner */
          ub[k]=k_ub/NN[k];
          k_ub=k_ub-ub[k]*NN[k];
          /* number of grid points in each dimension in the post */
          no[k]=(ub[k]-lb[k]+1);
          /* total no of post of (i,j) */
          npost*=no[k];
          cc[k]=0;

        }

        for(abs_type k=0; k<npost; k++) {
          abs_type q=0;
          for(int l=0; l<dim; l++) 
            q+=(lb[l]+cc[l])*NN[l];
          cc[0]++;
          for(int l=0; l<dim-1; l++) {
            if(cc[l]==no[l]) {
              cc[l]=0;
              cc[l+1]++;
            }
          }
          /* (i,j,q) is a transition */
          transition_function.m_pre[--transition_function.m_pre_ptr[q*M+j]]=i;
        }
      }
      /* print progress */
      if(m_verbose) {
        if(counter==0)
          std::cout << "2nd loop: ";
      }
      progress(i,N,counter);
    }
      std::cout << "total average time = " << ttimeavg/countavg << endl;    // mst 25 Jan18
  }
 
    //-------------------------------
    template<class F3=decltype(params::avoid_abs)>
    void compute_gb2(TransitionFunction& transition_function, const double tau, F3& avoid=params::avoid_abs) {
        // MST: intersection after taking interval hull of the zonotope
        
        /* number of cells */
        abs_type N=m_state_alphabet.size();
        //abs_type N=50;
        /* number of inputs */
        abs_type M=m_input_alphabet.size();
        //abs_type M=1;
        /* number of transitions (to be computed) */
        abs_ptr_type T=0;
        /* state space dimension */
        int dim=m_state_alphabet.get_dim();
        int dimInput = m_input_alphabet.get_dim();
        
        /* for display purpose */
        abs_type counter=0;
        /* some grid information */
        std::vector<abs_type> NN=m_state_alphabet.get_nn();
        /* variables for managing the post */
        std::vector<abs_type> lb(dim);  /* lower-left corner */
        std::vector<abs_type> ub(dim);  /* upper-right corner */
        std::vector<abs_type> no(dim);  /* number of cells per dim */
        std::vector<abs_type> cc(dim);  /* coordinate of current cell in the post */
        /* radius of hyper interval containing the attainable set */
        state_type eta;
        state_type r;
        /* state and input variables */
        state_type x;
        input_type u;
        /* for out of bounds check */
        state_type lower_left;
        state_type upper_right;
        /* copy data from m_state_alphabet */
        for(int i=0; i<dim; i++) {
            eta[i]=m_state_alphabet.get_eta()[i];
            lower_left[i]=m_state_alphabet.get_lower_left()[i];
            upper_right[i]=m_state_alphabet.get_upper_right()[i];
        }
        
        /* init in transition_function the members no_pre, no_post, pre_ptr */
        transition_function.init_infrastructure(N,M);
        /* lower-left & upper-right corners of hyper rectangle of cells that cover attainable set */
        std::unique_ptr<abs_type[]> corner_IDs(new abs_type[N*M*2]());  // <--- ?mst why multiplication by 2
        /* is post of (i,j) out of domain ? */
        std::unique_ptr<bool[]> out_of_domain(new bool[N*M]());
        
//        std::forward_list<bool> not_a_post;    // mst
        
        /*
         * first loop: compute corner_IDs:
         * corner_IDs[i*M+j][0] = lower-left cell index of over-approximation of attainable set
         * corner_IDs[i*M+j][1] = upper-right cell index of over-approximation of attainable set
         */
        /* loop over all cells */
        std::vector<double> L_hat_storage;   //-----------------
        L_hat_previous = Eigen::VectorXd::Zero(dim);    // 06 March 18
        
//        mstom::zonotope Z;
//        int count_iteration = 0;
        for(abs_type i=0; i<N; i++) { //N
            //std::cout<<"cell no = " << i << endl;
            /* is i an element of the avoid symbols ? */
            if(avoid(i)) {
                for(abs_type j=0; j<M; j++) {
                    out_of_domain[i*M+j]=true;
                }
                continue;
            }
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) { //M
//                std::cout << "j= "<< j << endl;
                out_of_domain[i*M+j]=false;
                /* get center x of cell */
                m_state_alphabet.itox(i,x);
                /* cell radius (including measurement errors) */
                for(int k=0; k<dim; k++)
                    r[k]=eta[k]/2.0+m_z[k];
                /* current input */
                m_input_alphabet.itox(j,u);
                /* integrate system and radius growth bound */
                /* the result is stored in x and r */
                //radius_post(r,x,u);
                //system_post(x,u);   // <---------- mst
                
//                ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
//                mstom::zonotope Z = ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
                countavg++;
                TicToc timet;
                timet.tic();
                ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage);
                ttimeavg += timet.tocMST();
                
                /* determine the cells which intersect with the attainable set:
                 * discrete hyper interval of cell indices
                 * [lb[0]; ub[0]] x .... x [lb[dim-1]; ub[dim-1]]
                 * covers attainable set
                 */
//                abs_type npostTrue; // mst
                abs_type npost=1;
                for(int k=0; k<dim; k++) {
                    /* check for out of bounds */
                    double left = x[k]-r[k]-m_z[k];
                    double right = x[k]+r[k]+m_z[k];
                    if(left <= lower_left[k]-eta[k]/2.0  || right >= upper_right[k]+eta[k]/2.0)  {
                        out_of_domain[i*M+j]=true;
                        break;
                    }
                    
                    /* integer coordinate of lower left corner of post */
                    lb[k] = static_cast<abs_type>((left-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* integer coordinate of upper right corner of post */
                    ub[k] = static_cast<abs_type>((right-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* number of grid points in the post in each dimension */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total number of post */
                    npost*=no[k];
                    cc[k]=0;
                }
//                npostTrue = npost;    // mst
//                cout<< "\nnpost, npostTrue = "<< npost << "  " << npostTrue << endl;
                corner_IDs[i*(2*M)+2*j]=0;
                corner_IDs[i*(2*M)+2*j+1]=0;
                if(out_of_domain[i*M+j])
                    continue;
                
                /* compute indices of post */
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    /* store id's of lower-left and upper-right cell */     // mst: shifted few lines up
                    if(k==0)
                        corner_IDs[i*(2*M)+2*j]=q;
                    if(k==npost-1)
                        corner_IDs[i*(2*M)+2*j+1]=q;
                    
//                    m_state_alphabet.itox(q,x); //mst
//                    mstom::zonotope Zcell = mstom::zonotope(x,eta,dim);
//                    mstom::zonotope Zsub = Z - Zcell;
//                    //ifOriginInZonotope(Zsub);
//                    tribool tb = ifOriginInZonotope(Zsub);
//
//                    std::vector<mstom::zonotope> plotsto;
//                    plotsto.push_back(Zcell);
//                    plotsto.push_back(Z);
//                    //plotsto.push_back(Zsub);
                    //plot(plotsto, 1, 2);
                    //cout << "\ntb = "<< tb << ", Is tb == indetermin: "<< (tb != true && tb != false) << endl;
//                    if(tb == false)
//                    {
//                        not_a_post.push_front(true) ;
//                        npostTrue = npostTrue - 1 ;
//                        continue;
//                    }
//                    else if(tb == true)
//                        not_a_post.push_front(false);
//                    else         // <-------------------------------
//                    {
//                        // indeterminate
//                     }
                    
                    /* (i,j,q) is a transition */
                    /* increment number of pres for (q,j) */
                     transition_function.m_no_pre[q*M+j]++;
//                    /* (i,j,q) is a transition */
//                    /* increment number of pres for (q,j) */
//                    transition_function.m_no_pre[q*M+j]++;
//                    /* store id's of lower-left and upper-right cell */
//                    if(k==0)
//                        corner_IDs[i*(2*M)+2*j]=q;
//                    if(k==npost-1)
//                        corner_IDs[i*(2*M)+2*j+1]=q;
                }
//                if(npost != npostTrue)
//                    cout<< "\nnpost, npostTrue = "<< npost << "  " << npostTrue << endl;
                /* increment number of transitions by number of post */
                T+=npost;
//                T+=npostTrue;   //mst
                transition_function.m_no_post[i*M+j]=npost;
//                transition_function.m_no_post[i*M+j]=npostTrue;   // mst
                
//                if(j==50)
//                    break;  //----------------
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "1st loop: ";
            }
            progress(i,N,counter);
            
//            if(i==10)
//                break;
        }
//        not_a_post.reverse();
        
//        mstom::plot(L_hat_storage);
        
        
        /* compute pre_ptr */
        abs_ptr_type sum=0;
        for(abs_type i=0; i<N; i++) {
            for(abs_type j=0; j<M; j++) {
                sum+=transition_function.m_no_pre[i*M+j];   // mst: ??----------
                transition_function.m_pre_ptr[i*M+j]=sum;   // mst: ??----------
            }
        }
        /* allocate memory for pre list */
        transition_function.init_transitions(T);
        
        /* second loop: fill pre array */
        counter=0;
        for(abs_type i=0; i<N; i++) {
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) {
                /* is x an element of the overflow symbols ? */
                if(out_of_domain[i*M+j])
                    continue;
                /* extract lower-left and upper-bound points */
                abs_type k_lb=corner_IDs[i*2*M+2*j];
                abs_type k_ub=corner_IDs[i*2*M+2*j+1];
                abs_type npost=1;
                
                /* cell idx to coordinates */
                for(int k=dim-1; k>=0; k--) {
                    /* integer coordinate of lower left corner */
                    lb[k]=k_lb/NN[k];
                    k_lb=k_lb-lb[k]*NN[k];
                    /* integer coordinate of upper right corner */
                    ub[k]=k_ub/NN[k];
                    k_ub=k_ub-ub[k]*NN[k];
                    /* number of grid points in each dimension in the post */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total no of post of (i,j) */
                    npost*=no[k];
                    cc[k]=0;
                    
                }
                
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    
//                    bool valA = not_a_post.front(); // mst
//                    not_a_post.pop_front();
//                    if(valA)    // not_a_post is true => q is not a post
//                        continue;
                    
                    /* (i,j,q) is a transition */
                    transition_function.m_pre[--transition_function.m_pre_ptr[q*M+j]]=i;    // mst: ??
                }
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "2nd loop: ";
            }
            progress(i,N,counter);
        }
        std::cout << "total average time = " << ttimeavg/countavg << endl;
        std::cout << "L_hat average time = " << LHatTimeavg/countavg << endl;
        std:: cout << "% of time for L_hat computation = " << LHatTimeavg * 100 / ttimeavg << endl;
    }
    
    //---------
    
    template<class F1, class F2, class F4, class F3=decltype(params::avoid_abs)>
    void compute_gb5(TransitionFunction& transition_function, const double tau,F1& system_post,
                     F2& radius_post,   F4& funcExpre, F3& avoid=params::avoid_abs) {
//         to plot both growth bound and zonotope sets
        // MST: intersection after taking interval hull of the zonotope
        
        /* number of cells */
        abs_type N=m_state_alphabet.size();
        //abs_type N=50;
        /* number of inputs */
        abs_type M=m_input_alphabet.size();
        //abs_type M=1;
        /* number of transitions (to be computed) */
        abs_ptr_type T=0;
        /* state space dimension */
        int dim=m_state_alphabet.get_dim();
        int dimInput = m_input_alphabet.get_dim();
        
        /* for display purpose */
        abs_type counter=0;
        /* some grid information */
        std::vector<abs_type> NN=m_state_alphabet.get_nn();
        /* variables for managing the post */
        std::vector<abs_type> lb(dim);  /* lower-left corner */
        std::vector<abs_type> ub(dim);  /* upper-right corner */
        std::vector<abs_type> no(dim);  /* number of cells per dim */
        std::vector<abs_type> cc(dim);  /* coordinate of current cell in the post */
        /* radius of hyper interval containing the attainable set */
        state_type eta;
        state_type r, r2, r2_eta, r_eta;
        /* state and input variables */
        state_type x, x2;
        input_type u;
        /* for out of bounds check */
        state_type lower_left;
        state_type upper_right;
        /* copy data from m_state_alphabet */
        for(int i=0; i<dim; i++) {
            eta[i]=m_state_alphabet.get_eta()[i];
            lower_left[i]=m_state_alphabet.get_lower_left()[i];
            upper_right[i]=m_state_alphabet.get_upper_right()[i];
        }
        
        /* init in transition_function the members no_pre, no_post, pre_ptr */
        transition_function.init_infrastructure(N,M);
        /* lower-left & upper-right corners of hyper rectangle of cells that cover attainable set */
        std::unique_ptr<abs_type[]> corner_IDs(new abs_type[N*M*2]());  // <--- ?mst why multiplication by 2
        /* is post of (i,j) out of domain ? */
        std::unique_ptr<bool[]> out_of_domain(new bool[N*M]());
        
        //        std::forward_list<bool> not_a_post;    // mst
        
        /*
         * first loop: compute corner_IDs:
         * corner_IDs[i*M+j][0] = lower-left cell index of over-approximation of attainable set
         * corner_IDs[i*M+j][1] = upper-right cell index of over-approximation of attainable set
         */
        /* loop over all cells */
        std::vector<double> L_hat_storage;   //-----------------
        //        mstom::zonotope Z;
        //        int count_iteration = 0;
        for(abs_type i=0; i<N; i++) { //N
            //std::cout<<"cell no = " << i << endl;
            /* is i an element of the avoid symbols ? */
            if(avoid(i)) {
                for(abs_type j=0; j<M; j++) {
                    out_of_domain[i*M+j]=true;
                }
                continue;
            }
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) { //M
                //                std::cout << "j= "<< j << endl;
                out_of_domain[i*M+j]=false;
                /* get center x of cell */
                m_state_alphabet.itox(i,x);
                /* cell radius (including measurement errors) */
                for(int k=0; k<dim; k++)
                    r[k]=eta[k]/2.0+m_z[k];
                /* current input */
                m_input_alphabet.itox(j,u);
                /* integrate system and radius growth bound */
                /* the result is stored in x and r */
                
                std::cout << "\nx: ";
                for(i=0;i<dim;i++)
                    std::cout << x[i] << ", ";
                std::cout << "\nu: ";
                for(i=0;i<dimInput ;i++)
                    std::cout << u[i] << ", ";
                
                
                r2 = r;
                x2 = x;
                radius_post(r2,x2,u);
                system_post(x2,u);
                
                //                mstom::zonotope Z = ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
                countavg++;
                TicToc timet;
                timet.tic();
                mstom::zonotope Zzo = ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
                ttimeavg += timet.tocMST();
                
//                for(i=0;i<dim;i++)
//                {
//                    std::cout<< "\nr, r2, x, x2:  " << r[i]<<", "<<r2[i] << ", " << x[i] << ", " << x2[i] << endl;
//                }
                
                for(int i=0;i<dim;i++)
                {
                    r2_eta[i] = 2 * r2[i];
                    r_eta[i] = 2 * r[i];
                }
                mstom::zonotope Zgb = mstom::zonotope(x2, r2_eta, dim);
                mstom::zonotope ZzoIH = mstom::zonotope(x, r_eta, dim);
                std::vector<mstom::zonotope> plotsto;
                plotsto.push_back(Zzo);
                plotsto.push_back(Zgb);
                plot(plotsto,1,2);
                
                
                
                
                
                /* determine the cells which intersect with the attainable set:
                 * discrete hyper interval of cell indices
                 * [lb[0]; ub[0]] x .... x [lb[dim-1]; ub[dim-1]]
                 * covers attainable set
                 */
                //                abs_type npostTrue; // mst
                abs_type npost=1;
                for(int k=0; k<dim; k++) {
                    /* check for out of bounds */
                    double left = x[k]-r[k]-m_z[k];
                    double right = x[k]+r[k]+m_z[k];
                    if(left <= lower_left[k]-eta[k]/2.0  || right >= upper_right[k]+eta[k]/2.0)  {
                        out_of_domain[i*M+j]=true;
                        break;
                    }
                    
                    /* integer coordinate of lower left corner of post */
                    lb[k] = static_cast<abs_type>((left-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* integer coordinate of upper right corner of post */
                    ub[k] = static_cast<abs_type>((right-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* number of grid points in the post in each dimension */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total number of post */
                    npost*=no[k];
                    cc[k]=0;
                }
                //                npostTrue = npost;    // mst
                //                cout<< "\nnpost, npostTrue = "<< npost << "  " << npostTrue << endl;
                corner_IDs[i*(2*M)+2*j]=0;
                corner_IDs[i*(2*M)+2*j+1]=0;
                if(out_of_domain[i*M+j])
                    continue;
                
                /* compute indices of post */
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    /* store id's of lower-left and upper-right cell */     // mst: shifted few lines up
                    if(k==0)
                        corner_IDs[i*(2*M)+2*j]=q;
                    if(k==npost-1)
                        corner_IDs[i*(2*M)+2*j+1]=q;
                    
                    //                    m_state_alphabet.itox(q,x); //mst
                    //                    mstom::zonotope Zcell = mstom::zonotope(x,eta,dim);
                    //                    mstom::zonotope Zsub = Z - Zcell;
                    //                    //ifOriginInZonotope(Zsub);
                    //                    tribool tb = ifOriginInZonotope(Zsub);
                    //
                    //                    std::vector<mstom::zonotope> plotsto;
                    //                    plotsto.push_back(Zcell);
                    //                    plotsto.push_back(Z);
                    //                    //plotsto.push_back(Zsub);
                    //plot(plotsto, 1, 2);
                    //cout << "\ntb = "<< tb << ", Is tb == indetermin: "<< (tb != true && tb != false) << endl;
                    //                    if(tb == false)
                    //                    {
                    //                        not_a_post.push_front(true) ;
                    //                        npostTrue = npostTrue - 1 ;
                    //                        continue;
                    //                    }
                    //                    else if(tb == true)
                    //                        not_a_post.push_front(false);
                    //                    else         // <-------------------------------
                    //                    {
                    //                        // indeterminate
                    //                     }
                    
                    /* (i,j,q) is a transition */
                    /* increment number of pres for (q,j) */
                    transition_function.m_no_pre[q*M+j]++;
                    //                    /* (i,j,q) is a transition */
                    //                    /* increment number of pres for (q,j) */
                    //                    transition_function.m_no_pre[q*M+j]++;
                    //                    /* store id's of lower-left and upper-right cell */
                    //                    if(k==0)
                    //                        corner_IDs[i*(2*M)+2*j]=q;
                    //                    if(k==npost-1)
                    //                        corner_IDs[i*(2*M)+2*j+1]=q;
                }
                //                if(npost != npostTrue)
                //                    cout<< "\nnpost, npostTrue = "<< npost << "  " << npostTrue << endl;
                /* increment number of transitions by number of post */
                T+=npost;
                //                T+=npostTrue;   //mst
                transition_function.m_no_post[i*M+j]=npost;
                //                transition_function.m_no_post[i*M+j]=npostTrue;   // mst
                
                //                if(j==50)
                //                    break;  //----------------
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "1st loop: ";
            }
            progress(i,N,counter);
            
            //            if(i==10)
            //                break;
        }
        //        not_a_post.reverse();
        
        //mstom::plot(L_hat_storage);
        
        
        /* compute pre_ptr */
        abs_ptr_type sum=0;
        for(abs_type i=0; i<N; i++) {
            for(abs_type j=0; j<M; j++) {
                sum+=transition_function.m_no_pre[i*M+j];   // mst: ??----------
                transition_function.m_pre_ptr[i*M+j]=sum;   // mst: ??----------
            }
        }
        /* allocate memory for pre list */
        transition_function.init_transitions(T);
        
        /* second loop: fill pre array */
        counter=0;
        for(abs_type i=0; i<N; i++) {
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) {
                /* is x an element of the overflow symbols ? */
                if(out_of_domain[i*M+j])
                    continue;
                /* extract lower-left and upper-bound points */
                abs_type k_lb=corner_IDs[i*2*M+2*j];
                abs_type k_ub=corner_IDs[i*2*M+2*j+1];
                abs_type npost=1;
                
                /* cell idx to coordinates */
                for(int k=dim-1; k>=0; k--) {
                    /* integer coordinate of lower left corner */
                    lb[k]=k_lb/NN[k];
                    k_lb=k_lb-lb[k]*NN[k];
                    /* integer coordinate of upper right corner */
                    ub[k]=k_ub/NN[k];
                    k_ub=k_ub-ub[k]*NN[k];
                    /* number of grid points in each dimension in the post */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total no of post of (i,j) */
                    npost*=no[k];
                    cc[k]=0;
                    
                }
                
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    
                    //                    bool valA = not_a_post.front(); // mst
                    //                    not_a_post.pop_front();
                    //                    if(valA)    // not_a_post is true => q is not a post
                    //                        continue;
                    
                    /* (i,j,q) is a transition */
                    transition_function.m_pre[--transition_function.m_pre_ptr[q*M+j]]=i;    // mst: ??
                }
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "2nd loop: ";
            }
            progress(i,N,counter);
        }
        std::cout << "total average time = " << ttimeavg/countavg << endl;
        std::cout << "L_hat average time = " << LHatTimeavg/countavg << endl;
        std:: cout << "% of time for L_hat computation = " << LHatTimeavg * 100 / ttimeavg << endl;
    }
    
    template<class F4, class F3=decltype(params::avoid_abs)>
    void compute_gb4(TransitionFunction& transition_function, const double tau, F4& funcExpre, F3& avoid=params::avoid_abs) {
        // MST: global Kprime
        
        /* number of cells */
        abs_type N=m_state_alphabet.size();
        //abs_type N=50;
        /* number of inputs */
        abs_type M=m_input_alphabet.size();
        //abs_type M=1;
        /* number of transitions (to be computed) */
        abs_ptr_type T=0;
        /* state space dimension */
        int dim=m_state_alphabet.get_dim();
        int dimInput = m_input_alphabet.get_dim();
        
        /* for display purpose */
        abs_type counter=0;
        /* some grid information */
        std::vector<abs_type> NN=m_state_alphabet.get_nn();
        /* variables for managing the post */
        std::vector<abs_type> lb(dim);  /* lower-left corner */
        std::vector<abs_type> ub(dim);  /* upper-right corner */
        std::vector<abs_type> no(dim);  /* number of cells per dim */
        std::vector<abs_type> cc(dim);  /* coordinate of current cell in the post */
        /* radius of hyper interval containing the attainable set */
        state_type eta;
        state_type r;
        /* state and input variables */
        state_type x;
        input_type u;
        /* for out of bounds check */
        state_type lower_left;
        state_type upper_right;
        /* copy data from m_state_alphabet */
        for(int i=0; i<dim; i++) {
            eta[i]=m_state_alphabet.get_eta()[i];
            lower_left[i]=m_state_alphabet.get_lower_left()[i];
            upper_right[i]=m_state_alphabet.get_upper_right()[i];
        }
        
        //12 Jan18, for global L_hat
        state_type inp_lower_left, inp_upper_right;
        for(int i=0; i<dimInput; i++) {
            inp_lower_left[i]=m_input_alphabet.get_lower_left()[i];
            inp_upper_right[i]=m_input_alphabet.get_upper_right()[i];
        }
        // MZ_L_hat: the zonotope over which hessian to be computed
        Eigen::VectorXd gL_hat = computeM(tau, lower_left, upper_right, inp_lower_left, inp_upper_right);
        cout << "\ngL_hat \n" << gL_hat << endl << endl;
        
        
        /* init in transition_function the members no_pre, no_post, pre_ptr */
        transition_function.init_infrastructure(N,M);
        /* lower-left & upper-right corners of hyper rectangle of cells that cover attainable set */
        std::unique_ptr<abs_type[]> corner_IDs(new abs_type[N*M*2]());  // <--- ?mst why multiplication by 2
        /* is post of (i,j) out of domain ? */
        std::unique_ptr<bool[]> out_of_domain(new bool[N*M]());
        
        //        std::forward_list<bool> not_a_post;    // mst
        
        /*
         * first loop: compute corner_IDs:
         * corner_IDs[i*M+j][0] = lower-left cell index of over-approximation of attainable set
         * corner_IDs[i*M+j][1] = upper-right cell index of over-approximation of attainable set
         */
        /* loop over all cells */
        std::vector<double> L_hat_storage;   //-----------------
        //        mstom::zonotope Z;
        //        int count_iteration = 0;
        for(abs_type i=0; i<N; i++) { //N
            //std::cout<<"cell no = " << i << endl;
            /* is i an element of the avoid symbols ? */
            if(avoid(i)) {
                for(abs_type j=0; j<M; j++) {
                    out_of_domain[i*M+j]=true;
                }
                continue;
            }
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) { //M
                //                std::cout << "j= "<< j << endl;
                out_of_domain[i*M+j]=false;
                /* get center x of cell */
                m_state_alphabet.itox(i,x);
                /* cell radius (including measurement errors) */
                for(int k=0; k<dim; k++)
                    r[k]=eta[k]/2.0+m_z[k];
                /* current input */
                m_input_alphabet.itox(j,u);
                /* integrate system and radius growth bound */
                /* the result is stored in x and r */
                //radius_post(r,x,u);
                //system_post(x,u);   // <---------- mst
                
                //                ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
                //                mstom::zonotope Z = ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
                countavg++;
                TicToc timet;
                timet.tic();
                ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
                ttimeavg += timet.tocMST();
                /* determine the cells which intersect with the attainable set:
                 * discrete hyper interval of cell indices
                 * [lb[0]; ub[0]] x .... x [lb[dim-1]; ub[dim-1]]
                 * covers attainable set
                 */
                //                abs_type npostTrue; // mst
                abs_type npost=1;
                for(int k=0; k<dim; k++) {
                    /* check for out of bounds */
                    double left = x[k]-r[k]-m_z[k];
                    double right = x[k]+r[k]+m_z[k];
                    if(left <= lower_left[k]-eta[k]/2.0  || right >= upper_right[k]+eta[k]/2.0)  {
                        out_of_domain[i*M+j]=true;
                        break;
                    }
                    
                    /* integer coordinate of lower left corner of post */
                    lb[k] = static_cast<abs_type>((left-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* integer coordinate of upper right corner of post */
                    ub[k] = static_cast<abs_type>((right-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* number of grid points in the post in each dimension */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total number of post */
                    npost*=no[k];
                    cc[k]=0;
                }
                //                npostTrue = npost;    // mst
                //                cout<< "\nnpost, npostTrue = "<< npost << "  " << npostTrue << endl;
                corner_IDs[i*(2*M)+2*j]=0;
                corner_IDs[i*(2*M)+2*j+1]=0;
                if(out_of_domain[i*M+j])
                    continue;
                
                /* compute indices of post */
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    /* store id's of lower-left and upper-right cell */     // mst: shifted few lines up
                    if(k==0)
                        corner_IDs[i*(2*M)+2*j]=q;
                    if(k==npost-1)
                        corner_IDs[i*(2*M)+2*j+1]=q;
                    
                    //                    m_state_alphabet.itox(q,x); //mst
                    //                    mstom::zonotope Zcell = mstom::zonotope(x,eta,dim);
                    //                    mstom::zonotope Zsub = Z - Zcell;
                    //                    //ifOriginInZonotope(Zsub);
                    //                    tribool tb = ifOriginInZonotope(Zsub);
                    //
                    //                    std::vector<mstom::zonotope> plotsto;
                    //                    plotsto.push_back(Zcell);
                    //                    plotsto.push_back(Z);
                    //                    //plotsto.push_back(Zsub);
                    //plot(plotsto, 1, 2);
                    //cout << "\ntb = "<< tb << ", Is tb == indetermin: "<< (tb != true && tb != false) << endl;
                    //                    if(tb == false)
                    //                    {
                    //                        not_a_post.push_front(true) ;
                    //                        npostTrue = npostTrue - 1 ;
                    //                        continue;
                    //                    }
                    //                    else if(tb == true)
                    //                        not_a_post.push_front(false);
                    //                    else         // <-------------------------------
                    //                    {
                    //                        // indeterminate
                    //                     }
                    
                    /* (i,j,q) is a transition */
                    /* increment number of pres for (q,j) */
                    transition_function.m_no_pre[q*M+j]++;
                    //                    /* (i,j,q) is a transition */
                    //                    /* increment number of pres for (q,j) */
                    //                    transition_function.m_no_pre[q*M+j]++;
                    //                    /* store id's of lower-left and upper-right cell */
                    //                    if(k==0)
                    //                        corner_IDs[i*(2*M)+2*j]=q;
                    //                    if(k==npost-1)
                    //                        corner_IDs[i*(2*M)+2*j+1]=q;
                }
                //                if(npost != npostTrue)
                //                    cout<< "\nnpost, npostTrue = "<< npost << "  " << npostTrue << endl;
                /* increment number of transitions by number of post */
                T+=npost;
                //                T+=npostTrue;   //mst
                transition_function.m_no_post[i*M+j]=npost;
                //                transition_function.m_no_post[i*M+j]=npostTrue;   // mst
                
                //                if(j==50)
                //                    break;  //----------------
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "1st loop: ";
            }
            progress(i,N,counter);
            
            //            if(i==10)
            //                break;
        }
        //        not_a_post.reverse();
        
        //mstom::plot(L_hat_storage);
        
        
        /* compute pre_ptr */
        abs_ptr_type sum=0;
        for(abs_type i=0; i<N; i++) {
            for(abs_type j=0; j<M; j++) {
                sum+=transition_function.m_no_pre[i*M+j];   // mst: ??----------
                transition_function.m_pre_ptr[i*M+j]=sum;   // mst: ??----------
            }
        }
        /* allocate memory for pre list */
        transition_function.init_transitions(T);
        
        /* second loop: fill pre array */
        counter=0;
        for(abs_type i=0; i<N; i++) {
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) {
                /* is x an element of the overflow symbols ? */
                if(out_of_domain[i*M+j])
                    continue;
                /* extract lower-left and upper-bound points */
                abs_type k_lb=corner_IDs[i*2*M+2*j];
                abs_type k_ub=corner_IDs[i*2*M+2*j+1];
                abs_type npost=1;
                
                /* cell idx to coordinates */
                for(int k=dim-1; k>=0; k--) {
                    /* integer coordinate of lower left corner */
                    lb[k]=k_lb/NN[k];
                    k_lb=k_lb-lb[k]*NN[k];
                    /* integer coordinate of upper right corner */
                    ub[k]=k_ub/NN[k];
                    k_ub=k_ub-ub[k]*NN[k];
                    /* number of grid points in each dimension in the post */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total no of post of (i,j) */
                    npost*=no[k];
                    cc[k]=0;
                    
                }
                
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    
                    //                    bool valA = not_a_post.front(); // mst
                    //                    not_a_post.pop_front();
                    //                    if(valA)    // not_a_post is true => q is not a post
                    //                        continue;
                    
                    /* (i,j,q) is a transition */
                    transition_function.m_pre[--transition_function.m_pre_ptr[q*M+j]]=i;    // mst: ??
                }
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "2nd loop: ";
            }
            progress(i,N,counter);
        }
        std::cout << "total average time = " << ttimeavg/countavg << endl;
        std::cout << "L_hat average time = " << LHatTimeavg/countavg << endl;
        std:: cout << "% of time for L_hat computation = " << LHatTimeavg * 100 / ttimeavg << endl;
    }
    
    template<class F4, class F3=decltype(params::avoid_abs)>
    void compute_gb3(TransitionFunction& transition_function, const double tau, F4& funcExpre, F3& avoid=params::avoid_abs) {
        // MST: to get intersection with zonotope
        /* number of cells */
        abs_type N=m_state_alphabet.size();
        //abs_type N=50;
        /* number of inputs */
        abs_type M=m_input_alphabet.size();
        //abs_type M=1;
        /* number of transitions (to be computed) */
        abs_ptr_type T=0;
        /* state space dimension */
        int dim=m_state_alphabet.get_dim();
        int dimInput = m_input_alphabet.get_dim();
        
        /* for display purpose */
        abs_type counter=0;
        /* some grid information */
        std::vector<abs_type> NN=m_state_alphabet.get_nn();
        /* variables for managing the post */
        std::vector<abs_type> lb(dim);  /* lower-left corner */
        std::vector<abs_type> ub(dim);  /* upper-right corner */
        std::vector<abs_type> no(dim);  /* number of cells per dim */
        std::vector<abs_type> cc(dim);  /* coordinate of current cell in the post */
        /* radius of hyper interval containing the attainable set */
        state_type eta;
        state_type r;
        /* state and input variables */
        state_type x;
        input_type u;
        /* for out of bounds check */
        state_type lower_left;
        state_type upper_right;
        /* copy data from m_state_alphabet */
        for(int i=0; i<dim; i++) {
            eta[i]=m_state_alphabet.get_eta()[i];
            lower_left[i]=m_state_alphabet.get_lower_left()[i];
            upper_right[i]=m_state_alphabet.get_upper_right()[i];
        }
        /* init in transition_function the members no_pre, no_post, pre_ptr */
        transition_function.init_infrastructure(N,M);
        /* lower-left & upper-right corners of hyper rectangle of cells that cover attainable set */
        std::unique_ptr<abs_type[]> corner_IDs(new abs_type[N*M*2]());  // <--- ?mst why multiplication by 2
        /* is post of (i,j) out of domain ? */
        std::unique_ptr<bool[]> out_of_domain(new bool[N*M]());
        
        std::forward_list<bool> not_a_post;    // mst
        
        /*
         * first loop: compute corner_IDs:
         * corner_IDs[i*M+j][0] = lower-left cell index of over-approximation of attainable set
         * corner_IDs[i*M+j][1] = upper-right cell index of over-approximation of attainable set
         */
        /* loop over all cells */
        std::vector<double> L_hat_storage;   //-----------------
        //        mstom::zonotope Z;
//        int count_iteration = 0;
        for(abs_type i=0; i<N; i++) { //N
            //std::cout<<"cell no = " << i << endl;
            /* is i an element of the avoid symbols ? */
            if(avoid(i)) {
                for(abs_type j=0; j<M; j++) {
                    out_of_domain[i*M+j]=true;
                }
                continue;
            }
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) { //M
                //                std::cout << "j= "<< j << endl;
                out_of_domain[i*M+j]=false;
                /* get center x of cell */
                m_state_alphabet.itox(i,x);
                /* cell radius (including measurement errors) */
                for(int k=0; k<dim; k++)
                    r[k]=eta[k]/2.0+m_z[k];
                /* current input */
                m_input_alphabet.itox(j,u);
                /* integrate system and radius growth bound */
                /* the result is stored in x and r */
                //radius_post(r,x,u);
                //system_post(x,u);   // <---------- mst
                
                //                ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
                mstom::zonotope Z = ReachableSet(dim, dimInput, tau, r, x, u, L_hat_storage, funcExpre);
                /* determine the cells which intersect with the attainable set:
                 * discrete hyper interval of cell indices
                 * [lb[0]; ub[0]] x .... x [lb[dim-1]; ub[dim-1]]
                 * covers attainable set
                 */
                abs_type npostTrue; // mst
                abs_type npost=1;
                for(int k=0; k<dim; k++) {
                    /* check for out of bounds */
                    double left = x[k]-r[k]-m_z[k];
                    double right = x[k]+r[k]+m_z[k];
                    if(left <= lower_left[k]-eta[k]/2.0  || right >= upper_right[k]+eta[k]/2.0)  {
                        out_of_domain[i*M+j]=true;
                        break;
                    }
                    
                    /* integer coordinate of lower left corner of post */
                    lb[k] = static_cast<abs_type>((left-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* integer coordinate of upper right corner of post */
                    ub[k] = static_cast<abs_type>((right-lower_left[k]+eta[k]/2.0)/eta[k]);
                    /* number of grid points in the post in each dimension */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total number of post */
                    npost*=no[k];
                    cc[k]=0;
                }
                npostTrue = npost;
                //                cout<< "\nnpost, npostTrue = "<< npost << "  " << npostTrue << endl;
                corner_IDs[i*(2*M)+2*j]=0;
                corner_IDs[i*(2*M)+2*j+1]=0;
                if(out_of_domain[i*M+j])
                    continue;
                
                /* compute indices of post */
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    /* store id's of lower-left and upper-right cell */     // mst: shifted few lines up
                    if(k==0)
                        corner_IDs[i*(2*M)+2*j]=q;
                    if(k==npost-1)
                        corner_IDs[i*(2*M)+2*j+1]=q;
                    
                    m_state_alphabet.itox(q,x); //mst
                    mstom::zonotope Zcell = mstom::zonotope(x,eta,dim);
                    mstom::zonotope Zsub = Z - Zcell;
//                    tribool tb = ifOriginInZonotope(Zsub);
                    bool tb = isOriginInZonotope(Zsub);
//                    cout << tb;
                    
//                    std::vector<mstom::zonotope> plotsto;
//                    plotsto.push_back(Zcell);
//                    plotsto.push_back(Z);
//                    //plotsto.push_back(Zsub);
//                    //plot(plotsto, 1, 2);
//                    plot(plotsto, 1, 2, tb);
                    
                    //cout << "\ntb = "<< tb << ", Is tb == indetermin: "<< (tb != true && tb != false) << endl;
                    if(tb == false)
                    {
                        
                        //cout << tb;
                        not_a_post.push_front(true) ;
                        npostTrue = npostTrue - 1 ;
                        continue;
                    }
                    else if(tb == true)
                        not_a_post.push_front(false);
                    else         // <-------------------------------
                    {
                        // indeterminate
                    }
                    
                    /* (i,j,q) is a transition */
                    /* increment number of pres for (q,j) */
                    transition_function.m_no_pre[q*M+j]++;
                    //                    /* (i,j,q) is a transition */
                    //                    /* increment number of pres for (q,j) */
                    //                    transition_function.m_no_pre[q*M+j]++;
                    //                    /* store id's of lower-left and upper-right cell */
                    //                    if(k==0)
                    //                        corner_IDs[i*(2*M)+2*j]=q;
                    //                    if(k==npost-1)
                    //                        corner_IDs[i*(2*M)+2*j+1]=q;
                }
                //                if(npost != npostTrue)
                //                    cout<< "\nnpost, npostTrue = "<< npost << "  " << npostTrue << endl;
                /* increment number of transitions by number of post */
                //                T+=npost;
                T+=npostTrue;   //mst
                //                transition_function.m_no_post[i*M+j]=npost;
                transition_function.m_no_post[i*M+j]=npostTrue;
                
                //                if(j==50)
                //                    break;  //----------------
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "1st loop: ";
            }
            progress(i,N,counter);
            
            //            if(i==10)
            //                break;
        }
        not_a_post.reverse();
        
        //mstom::plot(L_hat_storage);
        
        
        /* compute pre_ptr */
        abs_ptr_type sum=0;
        for(abs_type i=0; i<N; i++) {
            for(abs_type j=0; j<M; j++) {
                sum+=transition_function.m_no_pre[i*M+j];   // mst: ??----------
                transition_function.m_pre_ptr[i*M+j]=sum;   // mst: ??----------
            }
        }
        /* allocate memory for pre list */
        transition_function.init_transitions(T);
        
        /* second loop: fill pre array */
        counter=0;
        for(abs_type i=0; i<N; i++) {
            /* loop over all inputs */
            for(abs_type j=0; j<M; j++) {
                /* is x an element of the overflow symbols ? */
                if(out_of_domain[i*M+j])
                    continue;
                /* extract lower-left and upper-bound points */
                abs_type k_lb=corner_IDs[i*2*M+2*j];
                abs_type k_ub=corner_IDs[i*2*M+2*j+1];
                abs_type npost=1;
                
                /* cell idx to coordinates */
                for(int k=dim-1; k>=0; k--) {
                    /* integer coordinate of lower left corner */
                    lb[k]=k_lb/NN[k];
                    k_lb=k_lb-lb[k]*NN[k];
                    /* integer coordinate of upper right corner */
                    ub[k]=k_ub/NN[k];
                    k_ub=k_ub-ub[k]*NN[k];
                    /* number of grid points in each dimension in the post */
                    no[k]=(ub[k]-lb[k]+1);
                    /* total no of post of (i,j) */
                    npost*=no[k];
                    cc[k]=0;
                    
                }
                
                for(abs_type k=0; k<npost; k++) {
                    abs_type q=0;
                    for(int l=0; l<dim; l++)
                        q+=(lb[l]+cc[l])*NN[l];
                    cc[0]++;
                    for(int l=0; l<dim-1; l++) {
                        if(cc[l]==no[l]) {
                            cc[l]=0;
                            cc[l+1]++;
                        }
                    }
                    
                    bool valA = not_a_post.front(); // mst
                    not_a_post.pop_front();
                    if(valA)    // not_a_post is true => q is not a post
                        continue;
                    
                    /* (i,j,q) is a transition */
                    transition_function.m_pre[--transition_function.m_pre_ptr[q*M+j]]=i;    // mst: ??
                }
            }
            /* print progress */
            if(m_verbose) {
                if(counter==0)
                    std::cout << "2nd loop: ";
            }
            progress(i,N,counter);
        }
    }
    
    
    
    
    
    
    //--------------------------------
  /** @brief get the center of cells that are used to over-approximated the
   *  attainable set associated with cell (with center x) and input u
   *
   *  @returns std::vector<abs_type>
   *  @param system_post - lambda expression as in compute_gb
   *  @param radius_post - lambda expression as in compute_gb
   *  @param x - center of cell 
   *  @param u - input
   **/
  template<class F1, class F2>
  std::vector<state_type> get_post(F1& system_post,
                                   F2& radius_post,
                                   const state_type& x,
                                   const input_type& u) const {
    /* initialize return value */
    std::vector<state_type> post {};
    /* state space dimension */
    int dim = m_state_alphabet.get_dim();
    /* variables for managing the post */
    std::vector<abs_type> lb(dim);  /* lower-left corner */
    std::vector<abs_type> ub(dim);  /* upper-right corner */
    std::vector<abs_type> no(dim);  /* number of cells per dim */
    std::vector<abs_type> cc(dim);  /* coordinate of current cell in the post */
    /* get radius */
    state_type r;
    state_type eta;
    /* for out of bounds check */
    state_type lower_left;
    state_type upper_right;
    /* fill in data */
    for(int i=0; i<dim; i++) {
      eta[i]=m_state_alphabet.get_eta()[i];
      r[i]=m_state_alphabet.get_eta()[i]/2.0+m_z[i];
      lower_left[i]=m_state_alphabet.get_lower_left()[i];
      upper_right[i]=m_state_alphabet.get_upper_right()[i];
    }
    state_type xx=x;
    /* compute growth bound and numerical solution of ODE */
    radius_post(r,x,u);
    system_post(xx,u);

    /* determine post */
    abs_type npost=1;
    for(int k=0; k<dim; k++) {
      /* check for out of bounds */
      double left = xx[k]-r[k]-m_z[k];
      double right = xx[k]+r[k]+m_z[k];
      if(left <= lower_left[k]-eta[k]/2.0  || right >= upper_right[k]+eta[k]/2.0) {
        return post;
      } 
      /* integer coordinate of lower left corner of post */
      lb[k] = static_cast<abs_type>((left-lower_left[k]+eta[k]/2.0)/eta[k]);
      /* integer coordinate of upper right corner of post */
      ub[k] = static_cast<abs_type>((right-lower_left[k]+eta[k]/2.0)/eta[k]);
      /* number of grid points in the post in each dimension */
      no[k]=(ub[k]-lb[k]+1);
      /* total number of post */
      npost*=no[k];
      cc[k]=0;
    }
    /* compute indices of post */
    for(abs_type k=0; k<npost; k++) {
      abs_type q=0;
      for(int l=0; l<dim; l++)  {
        q+=(lb[l]+cc[l])*m_state_alphabet.get_nn()[l];
      }
      cc[0]++;
      for(int l=0; l<dim-1; l++) {
        if(cc[l]==no[l]) {
          cc[l]=0;
          cc[l+1]++;
        }
      }
      /* (i,j,q) is a transition */    
      m_state_alphabet.itox(q,xx);
      post.push_back(xx);
    }
    return post;
  }


  /** @brief print the center of cells that are stored in the transition
   *  function as post of the cell with center x and input u
   *
   *  @param transition_function - the transition function of the abstraction
   *  @param x - center of cell 
   *  @param u - input
   **/
  void print_post(const TransitionFunction& transition_function,
                  const state_type& x,
                  const input_type& u) const {
    std::vector<state_type> post = get_post(transition_function, x, u);
    if(!post.size())
      std::cout << "\nPost is out of domain\n";
    std::cout << "\nPost states: \n";
    for(abs_type v=0; v<post.size(); v++) {
      for(int i=0; i<m_state_alphabet.get_dim(); i++) {
        std::cout << post[v][i] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  /** @brief print the center of cells that are used to over-approximated the
   *  attainable set (computed according to the growth bound)
   *  associated with the cell with center x and input u
   *
   *  @param system_post - lambda expression as in compute_gb
   *  @param radius_post - lambda expression as in compute_gb
   *  @param x - center of cell 
   *  @param u - input
   **/
  template<class F1, class F2>
  void print_post_gb(F1& system_post,
                     F2& radius_post,
                     const state_type& x,
                     const input_type& u) const {
    std::vector<state_type> post = get_post(system_post,radius_post,x,u);
    if(!post.size()) {
      std::cout << "\nPost is out of domain\n";
      return;
    }

    std::cout << "\nPost states: \n";
    for(abs_type v=0; v<post.size(); v++) {
      for(int i=0; i<m_state_alphabet.get_dim(); i++) {
        std::cout << post[v][i] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  /** @brief set the measurement error bound **/
  void set_measurement_error_bound(const state_type& error_bound) {
    for(int i=0; i<m_state_alphabet.get_dim(); i++) {
      m_z[i]=error_bound[i];
    }
  }
  /** @brief get measurement error bound **/
  std::vector<double> get_measruement_error_bound() {
    std::vector<double> z;
    for(int i=0; i<m_state_alphabet.get_dim(); i++) {
      z.push_back(m_z[i]);
    }
    return z;
  }
  /** @brief activate console output **/
  void verbose_on() {
    m_verbose=true;
  }
  /** @brief deactivate console output **/
  void verbose_off() {
    m_verbose=false;
  }
};

} /* close namespace */
#endif /* ABSTRACTION_HH_ */
