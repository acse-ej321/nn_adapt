
#https://makefiletutorial.com/

all: uniform hessian go plot_qoi

.PHONY: test

ADAPTOR			= anisotropic isotropic hessian
INDICATOR       = mlp gnn
BASECASES       = ${offset} $(aligned)
TESTCASES       = $(normal) $(reverse) $(trench) $(viscosity_10)
OTHERCASES      = $(domain_wide) $(domain_long)  

aligned = "--coordinates 456,250.744,250 \
		   --casefolder "aligned" \
		   "

offset = "--coordinates 456,232.744,268 \
		  --casefolder "offset" \
		  "

normal = "--inflow_speed 5 \
		  --casefolder "normal" \
		   "

reverse = "--inflow_speed -5 \
		  --casefolder "reverse" \
		   "

viscosity_10 = "--viscosity_coefficient 10 \
		  		--casefolder "vis_10" \
				"

trench = "--inflow_speed 10 \
		  --viscosity_coefficient 2.0 \
		  --bathymetry_model trench\
		  --casefolder "trench" \
		  "
#"coordinates": [(L*0.5-144, W*0.5+18), (L*0.5+144, W*0.5-18)],
domain_wide = "--domain_length 1200 \
              --domain_width 1500 \
			  --infolw_speed 5 \
			  --coordinates 456,768 744,732 \
			  --casefolder "dom_wide" \
			  "

domain_long = "--domain_length 3600 \
              --domain_width 500 \
			  --infolw_speed 5 \
			  --coordinates 1656,268 1944,232 \
		      --casefolder "dom_long" \
			  "

testa: test1 test2

# Run full suite of test cases
# for 
run_convergence:
	for base in $(BASECASES); do \
		for case in $(TESTCASES); do \
			echo $$base $$case; \
			python3 turbine_run_forward.py \
			--adaptor_method "uniform" \
			$$base \
			$$case \
			--miniter 6 \
			--maxiter 8; \
			for adapt in $(ADAPTOR); do \
				echo $$adapt $$base $$case; \
				python3 turbine_run_convergence.py \
				--adaptor_method $$adapt \
				$$base \
				$$case \
				--miniter 3 \
				--maxiter 24; \
			done; \
		done; \
	done

test_casefolder_name:
	for base in $(BASECASES); do \
		for case in $(TESTCASES); do \
			python3 turbine_run_convergence.py \
			--adaptor_method hessian \
			$$case \
			$$base \
			--maxiter 2; \
		done; \
	done
# python3 turbine_run_forward.py \
# --adaptor_method "uniform" \
# --miniter 1 \
# --maxiter 2

# Convergence analysis for uniform refinement
# ===========================================
#
# Run the model on a sequence of uniformly refined meshes.

uniform:
	for case in $(TESTING_CASES); do \
		python3 turbine_run_forward.py \
		--adaptor_method $$case \
		--miniter 1 \
		--maxiter 2

hessian:
	python3 turbine_run_forward.py \
	--adaptor_method "hessian" \
	--miniter 3 \
	--maxiter 14

aniso:
	python3 turbine_run_forward.py \
	--adaptor_method "anisotropic" \
	--miniter 3 \
	--maxiter 14

mlp:
	python3 turbine_run_forward.py \
	--adaptor_method "anisotropic" \
	--indicator_method "mlp" \
	--miniter 3 \
	--maxiter 14

gnn:
	python3 turbine_run_forward.py \
	--adaptor_method "anisotropic" \
	--indicator_method "gnn" \
	--miniter 3 \
	--maxiter 14



plot:


clean:
