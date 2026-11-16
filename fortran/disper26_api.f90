! Owning interface to the common C++ API. Arrays use normal Fortran indexing;
! mode numbers remain zero-based. All operations require an api_status output.
! Failed handle-producing calls preserve existing output handles. Numeric
! extraction allocates arrays; raw pointers and manual releases stay private.
module disper26_api
  use, intrinsic :: iso_c_binding
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  implicit none
  private
  public :: c_double, material, model_geometry, computation_options, api_status
  public :: model, mode_solution, result_data, parameter_axis_entry, assignment(=)
  public :: create_isotropic_parameters, create_fluid_parameters, create_velocity_vti_parameters
  public :: create_layer_cake_model, create_pancake_model, create_grid_model
  public :: compute_dispersion, compute_dispersion_curve
  public :: compute_mode_count
  public :: create_model, dispersion_context, create_dispersion_context
  public :: compute_kernels, compute_hessian, compute_hessian_reduced, compute_interface_kernels
  public :: compute_interface_hessian_row, compute_eigenfunctions, compute_energy_integrals
  public :: compute_frequency_curvature, compute_gravity_correction
  public :: earth_flattening_transformation, apply_mineos_physical_dispersion, get_capabilities
  integer, parameter, public :: API_OK=0, API_INVALID_ARGUMENT=1, API_BUFFER_TOO_SMALL=2
  integer, parameter, public :: API_CALCULATION_FAILED=3, API_UNSUPPORTED=4, API_NOT_CONVERGED=5
  integer(c_size_t), parameter :: message_capacity=2048_c_size_t

  type, bind(C) :: material
    real(c_double) :: rho=0, A=0, C=0, F=0, L=0, N=0
    real(c_double) :: gravity=0, gravity_internal_scale=1
    real(c_double) :: gravity_surface_scale=0
  end type
  type :: model_geometry
    character(40) :: kind='flat'
    real(c_double) :: planet_radius=6371.0_c_double
    ! Explicit exponent is optional; an unset value uses the shared backend.
    real(c_double), allocatable :: flattening_exponent
  end type
  type, bind(C) :: c_geometry
    integer(c_int) :: kind_code
    real(c_double) :: planet_radius, flattening_exponent
  end type
  type :: computation_options
    integer :: max_steps=5, max_levels=5, num_sub_layers=10
    real(c_double) :: tolerance=1e-10_c_double, root_tolerance=1e-12_c_double
    real(c_double) :: derivative_tolerance=1e-6_c_double
    real(c_double), allocatable :: c_guess
    character(63) :: method='standard', backend='dual', integration='auto'
    logical :: strict=.false.
    character(63) :: start='long', predictor='auto', verify='', on_violation=''
    real(c_double), allocatable :: c_range(:)
    logical, allocatable :: verify_mode
  end type
  type, bind(C) :: c_options
    integer(c_int) :: max_steps, max_levels, num_sub_layers
    real(c_double) :: tolerance, root_tolerance, derivative_tolerance
    integer(c_int) :: has_c_guess
    real(c_double) :: c_guess
    character(kind=c_char) :: method(64), backend(64), integration(64)
    integer(c_int) :: strict
  end type
  type, bind(C) :: c_tracking_options
    integer(c_size_t) :: struct_size
    character(kind=c_char) :: start(64), predictor(64), verify(64), on_violation(64)
    integer(c_int) :: has_c_range
    real(c_double) :: c_min, c_max
    integer(c_int) :: verify_mode
  end type
  type :: api_status
    integer :: code=API_OK
    character(:), allocatable :: message
  contains
    procedure :: ok => status_ok
  end type
  type :: parameter_axis_entry
    character(:), allocatable :: name, entity, unit
    integer :: index=0 ! One-based layer/node/interface; mode is unaffected.
    logical :: active=.false.
  end type
  type :: dispersion_context
    private
    type(c_ptr) :: handle=c_null_ptr
  contains
    final :: finalize_context
    procedure :: valid => context_valid
  end type
  type :: model
    private
    type(c_ptr) :: handle=c_null_ptr
  contains
    final :: finalize_model
    procedure :: info => model_info
    procedure :: valid => model_valid
  end type
  type :: mode_solution
    private
    type(c_ptr) :: handle=c_null_ptr
  contains
    final :: finalize_solution
    procedure :: info => solution_info
    procedure :: get_model => solution_model
    procedure :: phase_velocity => solution_phase_velocity
    procedure :: group_velocity => solution_group_velocity
    procedure :: hv_ratio => solution_hv_ratio
    procedure :: valid => solution_valid
  end type
  type :: result_data
    private
    type(c_ptr) :: handle=c_null_ptr
  contains
    final :: finalize_result
    procedure :: numeric => result_numeric
    procedure :: vector => result_vector
    procedure :: scalar => result_scalar
    procedure :: text => result_text
    procedure :: field_names => result_field_names
    procedure :: parameter => result_parameter
    procedure :: get_solution => curve_solution
    procedure :: valid => result_valid
  end type
  interface assignment(=)
    module procedure assign_model, assign_solution, assign_result, assign_context
  end interface
  interface

    subroutine c_model_retain(handle) bind(C,name="d26_api_model_retain")
      import

      type(c_ptr), value :: handle

    end subroutine

    subroutine c_model_release(handle) bind(C,name="d26_api_model_release")
      import

      type(c_ptr), value :: handle

    end subroutine

    subroutine c_solution_retain(handle) bind(C,name="d26_api_solution_retain")
      import

      type(c_ptr), value :: handle

    end subroutine

    subroutine c_solution_release(handle) bind(C,name="d26_api_solution_release")
      import

      type(c_ptr), value :: handle

    end subroutine

    subroutine c_result_retain(handle) bind(C,name="d26_api_result_retain")
      import

      type(c_ptr), value :: handle

    end subroutine

    subroutine c_result_release(handle) bind(C,name="d26_api_result_release")
      import

      type(c_ptr), value :: handle

    end subroutine

    subroutine c_default_options(opts) bind(C,name="d26_api_default_options")
      import

      type(c_options), intent(out) :: opts

    end subroutine

    function c_isotropic(vs,vp,rho,out,msg,cap) &
        bind(C,name="d26_api_create_isotropic_parameters") result(code)
      import
      integer(c_int) :: code

      real(c_double), value :: vs,vp,rho

      type(material), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_fluid(vp,rho,out,msg,cap) &
        bind(C,name="d26_api_create_fluid_parameters") result(code)
      import
      integer(c_int) :: code

      real(c_double), value :: vp,rho

      type(material), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_velocity_vti(rho,vph,vpv,vsh,vsv,eta,out,msg,cap) &
        bind(C,name="d26_api_create_velocity_vti_parameters") result(code)
      import
      integer(c_int) :: code

      real(c_double), value :: rho,vph,vpv,vsh,vsv,eta

      type(material), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_create_model(n,depths,kind,basis,values,components,interpolation,geometry,name,out,msg,cap) &
        bind(C,name="d26_api_create_model") result(code)
      import
      integer(c_int) :: code
      integer(c_size_t), value :: n,components,cap
      real(c_double), intent(in) :: depths(*),values(*)
      character(kind=c_char), intent(in) :: kind(*),basis(*),name(*)
      type(c_ptr), value :: interpolation
      type(c_geometry), intent(in) :: geometry
      type(c_ptr), intent(out) :: out
      character(kind=c_char), intent(out) :: msg(*)
    end function

    function c_layer_cake(n,layers,thicknesses,halfspace,geometry,name,out,msg,cap) &
        bind(C,name="d26_api_create_layer_cake_model") result(code)
      import
      integer(c_int) :: code

      integer(c_size_t), value :: n

      type(material), intent(in) :: layers(*), halfspace

      real(c_double), intent(in) :: thicknesses(*)

      type(c_geometry), intent(in) :: geometry

      character(kind=c_char), intent(in) :: name(*)

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_pancake(n,layers,thicknesses,halfspace,geometry,name,out,msg,cap) &
        bind(C,name="d26_api_create_pancake_model") result(code)
      import
      integer(c_int) :: code

      integer(c_size_t), value :: n

      type(material), intent(in) :: layers(*), halfspace

      real(c_double), intent(in) :: thicknesses(*)

      type(c_geometry), intent(in) :: geometry

      character(kind=c_char), intent(in) :: name(*)

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_grid(n,depths,materials,interpolation,geometry,name,out,msg,cap) &
        bind(C,name="d26_api_create_grid_model") result(code)
      import
      integer(c_int) :: code

      integer(c_size_t), value :: n

      real(c_double), intent(in) :: depths(*)

      type(material), intent(in) :: materials(*)

      character(kind=c_char), intent(in) :: interpolation(*), name(*)

      type(c_geometry), intent(in) :: geometry

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    subroutine c_default_tracking_options(opts) bind(C,name="d26_api_default_tracking_options")
      import
      type(c_tracking_options), intent(out) :: opts
    end subroutine

    function c_dispersion_ex(handle,period,wave,mode,outputs,opts,tracking,context,out,msg,cap) &
        bind(C,name="d26_api_compute_dispersion_ex") result(code)
      import
      integer(c_int) :: code
      type(c_ptr), value :: handle, context
      real(c_double), value :: period
      integer(c_int), value :: mode
      character(kind=c_char), intent(in) :: wave(*), outputs(*)
      type(c_options), intent(in) :: opts
      type(c_tracking_options), intent(in) :: tracking
      type(c_ptr), intent(out) :: out
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_size_t), value :: cap
    end function

    function c_curve_ex(handle,n,periods,wave,mode,outputs,opts,tracking,context,out,msg,cap) &
        bind(C,name="d26_api_compute_dispersion_curve_ex") result(code)
      import
      integer(c_int) :: code
      type(c_ptr), value :: handle, context
      integer(c_size_t), value :: n
      real(c_double), intent(in) :: periods(*)
      integer(c_int), value :: mode
      character(kind=c_char), intent(in) :: wave(*), outputs(*)
      type(c_options), intent(in) :: opts
      type(c_tracking_options), intent(in) :: tracking
      type(c_ptr), intent(out) :: out
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_size_t), value :: cap
    end function

    function c_dispersion(handle,period,wave,mode,outputs,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_dispersion") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      real(c_double), value :: period

      integer(c_int), value :: mode

      character(kind=c_char), intent(in) :: wave(*), outputs(*)

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_curve(handle,n,periods,wave,mode,outputs,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_dispersion_curve") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      integer(c_size_t), value :: n

      real(c_double), intent(in) :: periods(*)

      integer(c_int), value :: mode

      character(kind=c_char), intent(in) :: wave(*), outputs(*)

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_dispersion_context(handle,period,wave,mode,outputs,opts,context,out,msg,cap) &
        bind(C,name="d26_api_compute_dispersion_with_context") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle, context

      real(c_double), value :: period

      integer(c_int), value :: mode

      character(kind=c_char), intent(in) :: wave(*), outputs(*)

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_curve_context(handle,n,periods,wave,mode,outputs,opts,context,out,msg,cap) &
        bind(C,name="d26_api_compute_dispersion_curve_with_context") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle, context

      integer(c_size_t), value :: n

      real(c_double), intent(in) :: periods(*)

      integer(c_int), value :: mode

      character(kind=c_char), intent(in) :: wave(*), outputs(*)

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_create_context(out,msg,cap) bind(C,name="d26_api_create_dispersion_context") result(code)
      import
      type(c_ptr), intent(out) :: out
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_size_t), value :: cap
      integer(c_int) :: code
    end function
    subroutine c_context_retain(handle) bind(C,name="d26_api_dispersion_context_retain")
      import
      type(c_ptr), value :: handle
    end subroutine
    subroutine c_context_release(handle) bind(C,name="d26_api_dispersion_context_release")
      import
      type(c_ptr), value :: handle
    end subroutine

    function c_model_info(handle,out,msg,cap) &
        bind(C,name="d26_api_model_info") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_mode_count(handle,level,count,msg,cap) &
        bind(C,name="d26_api_compute_mode_count") result(code)
      import
      integer(c_int) :: code
      type(c_ptr), value :: handle
      integer(c_int), value :: level
      integer(c_int), intent(out) :: count
      character(kind=c_char), intent(out) :: msg(*)
      integer(c_size_t), value :: cap
    end function

    function c_solution_info(handle,out,msg,cap) &
        bind(C,name="d26_api_solution_info") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_solution_model(handle,out,msg,cap) &
        bind(C,name="d26_api_solution_model") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_earth_flattening_transformation(handle,out,msg,cap) &
        bind(C,name="d26_api_earth_flattening_transformation") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_curve_solution(handle,index,out,msg,cap) &
        bind(C,name="d26_api_curve_solution") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      integer(c_size_t), value :: index

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_kernels(handle,basis,outputs,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_kernels") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      character(kind=c_char), intent(in) :: basis(*)

      character(kind=c_char), intent(in) :: outputs(*)

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_hessian(handle,basis,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_hessian") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      character(kind=c_char), intent(in) :: basis(*)

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_interface_hessian_row(handle,index,basis,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_interface_hessian_row") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      integer(c_int), value :: index

      character(kind=c_char), intent(in) :: basis(*)

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_interface_kernels(handle,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_interface_kernels") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_energy_integrals(handle,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_energy_integrals") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_frequency_curvature(handle,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_frequency_curvature") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_reduced(handle,projection,rows,columns,opts,out,msg,cap) &
        bind(C,name="d26_api_compute_hessian_reduced") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      real(c_double), intent(in) :: projection(*)

      integer(c_size_t), value :: rows,columns

      type(c_options), intent(in) :: opts

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_eigenfunctions(handle,n,out,msg,cap) &
        bind(C,name="d26_api_compute_eigenfunctions") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      integer(c_int), value :: n

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_gravity(handle,gravity,region,n,domega,out,msg,cap) &
        bind(C,name="d26_api_compute_gravity_correction") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      real(c_double), value :: gravity,domega

      character(kind=c_char), intent(in) :: region(*)

      integer(c_int), value :: n

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_q(handle,n,qk,qm,reference_period,period,out,msg,cap) &
        bind(C,name="d26_api_apply_mineos_physical_dispersion") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      integer(c_size_t), value :: n

      real(c_double), intent(in) :: qk(*),qm(*)

      real(c_double), value :: reference_period,period

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_capabilities(handle,wave,out,msg,cap) &
        bind(C,name="d26_api_get_capabilities") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      character(kind=c_char), intent(in) :: wave(*)

      type(c_ptr), intent(out) :: out

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_numeric(handle,field,values,capacity,rows,columns,msg,cap) &
        bind(C,name="d26_api_result_numeric") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle,values

      character(kind=c_char), intent(in) :: field(*)

      integer(c_size_t), value :: capacity

      integer(c_size_t), intent(out) :: rows,columns

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_text(handle,field,value,capacity,required,msg,cap) &
        bind(C,name="d26_api_result_string") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle,value

      character(kind=c_char), intent(in) :: field(*)

      integer(c_size_t), value :: capacity

      integer(c_size_t), intent(out) :: required

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_field_count(handle,numeric,count,msg,cap) &
        bind(C,name="d26_api_result_field_count") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle

      integer(c_int), value :: numeric

      integer(c_size_t), intent(out) :: count

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

    function c_field_name(handle,numeric,index,value,capacity,required,msg,cap) &
        bind(C,name="d26_api_result_field_name") result(code)
      import
      integer(c_int) :: code

      type(c_ptr), value :: handle,value

      integer(c_int), value :: numeric

      integer(c_size_t), value :: index,capacity

      integer(c_size_t), intent(out) :: required

      character(kind=c_char), intent(out) :: msg(*)

      integer(c_size_t), value :: cap

    end function

  end interface
contains
  logical function status_ok(this)
    class(api_status), intent(in) :: this
    status_ok=this%code==API_OK
  end function
  subroutine finish_status(status,code,msg)
    type(api_status), intent(out) :: status
    integer(c_int), intent(in) :: code
    character(kind=c_char), intent(in) :: msg(:)
    integer :: i,n
    status%code=code
    n=0
    do i=1,size(msg)
      if(msg(i)==c_null_char) exit
      n=n+1
    end do
    allocate(character(n)::status%message)
    do i=1,n
      status%message(i:i)=msg(i)
    end do
  end subroutine
  subroutine invalid(status,message)
    type(api_status), intent(out) :: status
    character(*), intent(in) :: message
    status%code=API_INVALID_ARGUMENT
    status%message=message
  end subroutine
  subroutine pack_string(input,output)
    character(*), intent(in) :: input
    character(kind=c_char), intent(out) :: output(:)
    integer :: i
    output=c_null_char
    do i=1,min(len_trim(input),size(output)-1)
      output(i)=input(i:i)
    end do
  end subroutine
  function pack_options(input) result(out)
    type(computation_options), optional, intent(in) :: input
    type(c_options) :: out
    call c_default_options(out)
    if(.not.present(input)) return
    out%max_steps=input%max_steps
    out%max_levels=input%max_levels
    out%num_sub_layers=input%num_sub_layers
    out%tolerance=input%tolerance
    out%root_tolerance=input%root_tolerance
    out%derivative_tolerance=input%derivative_tolerance
    out%has_c_guess=0
    if(allocated(input%c_guess)) then
      out%has_c_guess=1
      out%c_guess=input%c_guess
    end if
    call pack_string(input%method,out%method)
    call pack_string(input%backend,out%backend)
    call pack_string(input%integration,out%integration)
    out%strict=merge(1,0,input%strict)
  end function
  function pack_tracking_options(input,status) result(out)
    type(computation_options), optional, intent(in) :: input
    type(api_status), intent(out) :: status
    type(c_tracking_options) :: out
    call c_default_tracking_options(out)
    if(.not.present(input)) return
    call pack_string(input%start,out%start)
    call pack_string(input%predictor,out%predictor)
    call pack_string(input%verify,out%verify)
    call pack_string(input%on_violation,out%on_violation)
    if(allocated(input%c_range)) then
      if(size(input%c_range)/=2) then
        call invalid(status,'c_range must contain two velocities')
        return
      end if
      out%has_c_range=1
      out%c_min=input%c_range(1)
      out%c_max=input%c_range(2)
    end if
    if(allocated(input%verify_mode)) out%verify_mode=merge(1,0,input%verify_mode)
  end function
  subroutine pack_geometry(input,out,status)
    type(model_geometry), optional, intent(in) :: input
    type(c_geometry), intent(out) :: out
    type(api_status), intent(out) :: status
    out%kind_code=0
    out%planet_radius=6371.0_c_double
    out%flattening_exponent=ieee_value(0.0_c_double,ieee_quiet_nan)
    status%code=API_OK
    status%message=''
    if(.not.present(input)) return
    select case(trim(input%kind))
    case('flat')
      out%kind_code=0
    case('spherical_ba97')
      out%kind_code=1
    case('spherical_experimental_density')
      out%kind_code=2
    case default
      call invalid(status,'Geometry must be flat, spherical_ba97 or spherical_experimental_density')
      return
    end select
    out%planet_radius=input%planet_radius
    if(allocated(input%flattening_exponent)) out%flattening_exponent=input%flattening_exponent
  end subroutine

  impure elemental subroutine finalize_context(this)
    type(dispersion_context), intent(inout) :: this
    call c_context_release(this%handle)
    this%handle=c_null_ptr
  end subroutine
  impure elemental subroutine assign_context(left,right)
    type(dispersion_context), intent(inout) :: left
    type(dispersion_context), intent(in) :: right
    type(c_ptr) :: handle
    handle=right%handle
    call c_context_retain(handle)
    call c_context_release(left%handle)
    left%handle=handle
  end subroutine
  logical function context_valid(this)
    class(dispersion_context), intent(in) :: this
    context_valid=c_associated(this%handle)
  end function
  subroutine adopt_context(out,handle,status)
    type(dispersion_context), intent(inout) :: out
    type(c_ptr), intent(in) :: handle
    type(api_status), intent(in) :: status
    if(.not.status%ok()) return
    call c_context_release(out%handle)
    out%handle=handle
  end subroutine

  subroutine create_dispersion_context(out,status)
    type(dispersion_context), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_create_context(handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_context(out,handle,status)
  end subroutine

  impure elemental subroutine finalize_model(this)
    type(model), intent(inout) :: this
    call c_model_release(this%handle)
    this%handle=c_null_ptr
  end subroutine
  impure elemental subroutine assign_model(left,right)
    type(model), intent(inout) :: left
    type(model), intent(in) :: right
    type(c_ptr) :: handle
    handle=right%handle
    call c_model_retain(handle)
    call c_model_release(left%handle)
    left%handle=handle
  end subroutine
  logical function model_valid(this)
    class(model), intent(in) :: this
    model_valid=c_associated(this%handle)
  end function
  subroutine adopt_model(out,handle,status)
    type(model), intent(inout) :: out
    type(c_ptr), intent(in) :: handle
    type(api_status), intent(in) :: status
    if(.not.status%ok()) return
    call c_model_release(out%handle)
    out%handle=handle
  end subroutine

  impure elemental subroutine finalize_solution(this)
    type(mode_solution), intent(inout) :: this
    call c_solution_release(this%handle)
    this%handle=c_null_ptr
  end subroutine
  impure elemental subroutine assign_solution(left,right)
    type(mode_solution), intent(inout) :: left
    type(mode_solution), intent(in) :: right
    type(c_ptr) :: handle
    handle=right%handle
    call c_solution_retain(handle)
    call c_solution_release(left%handle)
    left%handle=handle
  end subroutine
  logical function solution_valid(this)
    class(mode_solution), intent(in) :: this
    solution_valid=c_associated(this%handle)
  end function
  subroutine adopt_solution(out,handle,status)
    type(mode_solution), intent(inout) :: out
    type(c_ptr), intent(in) :: handle
    type(api_status), intent(in) :: status
    if(.not.status%ok()) return
    call c_solution_release(out%handle)
    out%handle=handle
  end subroutine

  impure elemental subroutine finalize_result(this)
    type(result_data), intent(inout) :: this
    call c_result_release(this%handle)
    this%handle=c_null_ptr
  end subroutine
  impure elemental subroutine assign_result(left,right)
    type(result_data), intent(inout) :: left
    type(result_data), intent(in) :: right
    type(c_ptr) :: handle
    handle=right%handle
    call c_result_retain(handle)
    call c_result_release(left%handle)
    left%handle=handle
  end subroutine
  logical function result_valid(this)
    class(result_data), intent(in) :: this
    result_valid=c_associated(this%handle)
  end function
  subroutine adopt_result(out,handle,status)
    type(result_data), intent(inout) :: out
    type(c_ptr), intent(in) :: handle
    type(api_status), intent(in) :: status
    if(.not.status%ok()) return
    call c_result_release(out%handle)
    out%handle=handle
  end subroutine

  subroutine create_isotropic_parameters(vs,vp,rho,out,status)
    real(c_double), intent(in) :: vs,vp,rho
    type(material), intent(out) :: out
    type(api_status), intent(out) :: status
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_isotropic(vs,vp,rho,out,msg,message_capacity)
    call finish_status(status,code,msg)
  end subroutine

  subroutine create_fluid_parameters(vp,rho,out,status)
    real(c_double), intent(in) :: vp,rho
    type(material), intent(out) :: out
    type(api_status), intent(out) :: status
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_fluid(vp,rho,out,msg,message_capacity)
    call finish_status(status,code,msg)
  end subroutine

  subroutine create_velocity_vti_parameters(rho,vph,vpv,vsh,vsv,eta,out,status)
    real(c_double), intent(in) :: rho,vph,vpv,vsh,vsv,eta
    type(material), intent(out) :: out
    type(api_status), intent(out) :: status
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_velocity_vti(rho,vph,vpv,vsh,vsv,eta,out,msg,message_capacity)
    call finish_status(status,code,msg)
  end subroutine

  subroutine create_model(depths,kind,out,status,vs,vp,rho,vph,vpv,vsh,vsv,eta,A,C,F,L,N,interpolation,geometry,name)
    real(c_double), intent(in) :: depths(:)
    character(*), intent(in) :: kind
    type(model), intent(inout) :: out
    type(api_status), intent(out) :: status
    real(c_double), optional, intent(in) :: vs(:),vp(:),rho(:),vph(:),vpv(:),vsh(:),vsv(:),eta(:)
    real(c_double), optional, intent(in) :: A(:),C(:),F(:),L(:),N(:)
    character(*), optional, intent(in) :: interpolation,name
    type(model_geometry), optional, intent(in) :: geometry
    real(c_double), allocatable :: values(:,:)
    character(:), allocatable :: basis,actual_name
    character(kind=c_char), allocatable, target :: interp_chars(:)
    character(kind=c_char) :: msg(message_capacity)
    type(c_ptr) :: handle,interp_ptr
    type(c_geometry) :: geo
    integer(c_int) :: code
    integer :: components,i
    logical :: iso,velocity,stiffness
    status%code=API_OK
    status%message=''
    iso=present(vs).or.present(vp)
    velocity=present(vph).or.present(vpv).or.present(vsh).or.present(vsv).or.present(eta)
    stiffness=present(A).or.present(C).or.present(F).or.present(L).or.present(N)
    if(merge(1,0,iso)+merge(1,0,velocity)+merge(1,0,stiffness)/=1) then
      call invalid(status,'Supply exactly one complete material representation')
      return
    end if
    components=6
    if(iso) components=3
    allocate(values(components,size(depths)))
    if(iso) then
      basis='isotropic'
      call field(1,vs,'vs'); call field(2,vp,'vp'); call field(3,rho,'rho')
    else if(velocity) then
      basis='velocity_vti'
      call field(1,rho,'rho'); call field(2,vph,'vph'); call field(3,vpv,'vpv')
      call field(4,vsh,'vsh'); call field(5,vsv,'vsv'); call field(6,eta,'eta')
    else
      basis='stiffness'
      call field(1,A,'A'); call field(2,C,'C'); call field(3,F,'F')
      call field(4,L,'L'); call field(5,N,'N'); call field(6,rho,'rho')
    end if
    if(.not.status%ok()) return
    call pack_geometry(geometry,geo,status)
    if(.not.status%ok()) return
    actual_name=''
    if(present(name)) actual_name=name
    interp_ptr=c_null_ptr
    if(present(interpolation)) then
      allocate(interp_chars(len_trim(interpolation)+1))
      do i=1,len_trim(interpolation)
        interp_chars(i)=interpolation(i:i)
      end do
      interp_chars(size(interp_chars))=c_null_char
      interp_ptr=c_loc(interp_chars(1))
    end if
    code=c_create_model(size(depths,kind=c_size_t),depths,trim(kind)//c_null_char,basis//c_null_char, &
        values,int(components,c_size_t),interp_ptr,geo,actual_name//c_null_char,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_model(out,handle,status)
  contains
    subroutine field(row,x,label)
      integer, intent(in) :: row
      real(c_double), optional, intent(in) :: x(:)
      character(*), intent(in) :: label
      if(.not.status%ok()) return
      if(.not.present(x)) then
        call invalid(status,'Missing material array: '//label)
      else if(size(x)/=size(depths)) then
        call invalid(status,label//' must match depths length')
      else
        values(row,:)=x
      end if
    end subroutine
  end subroutine

  subroutine create_layer_cake_model(layers,thicknesses,halfspace,out,status,geometry,name)
    type(material), intent(in) :: layers(:),halfspace
    real(c_double), intent(in) :: thicknesses(:)
    type(model), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(model_geometry), optional, intent(in) :: geometry
    character(*), optional, intent(in) :: name
    call create_layer_cake_model_impl(layers,thicknesses,halfspace,out,status,geometry,name, &
        .false.)
  end subroutine
  subroutine create_pancake_model(layers,thicknesses,halfspace,out,status,geometry,name)
    type(material), intent(in) :: layers(:),halfspace
    real(c_double), intent(in) :: thicknesses(:)
    type(model), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(model_geometry), optional, intent(in) :: geometry
    character(*), optional, intent(in) :: name
    call create_layer_cake_model_impl(layers,thicknesses,halfspace,out,status,geometry,name, &
        .true.)
  end subroutine
  subroutine create_layer_cake_model_impl(layers,thicknesses,halfspace,out,status,geometry,name,legacy)
    type(material), intent(in) :: layers(:),halfspace
    real(c_double), intent(in) :: thicknesses(:)
    type(model), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(model_geometry), optional, intent(in) :: geometry
    character(*), optional, intent(in) :: name
    logical, intent(in) :: legacy
    character(:), allocatable :: actual_name
    type(c_geometry) :: geo
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    if(size(layers)/=size(thicknesses)) then
      call invalid(status,'layers and thicknesses must have equal lengths')
      return
    end if
    call pack_geometry(geometry,geo,status)
    if(.not.status%ok()) return
    actual_name=''
    if(present(name)) actual_name=name
    if(legacy) then
      code=c_pancake(size(layers,kind=c_size_t),layers,thicknesses,halfspace,geo, &
        actual_name//c_null_char,handle,msg,message_capacity)
    else
      code=c_layer_cake(size(layers,kind=c_size_t),layers,thicknesses,halfspace,geo, &
        actual_name//c_null_char,handle,msg,message_capacity)
    end if
    call finish_status(status,code,msg)
    call adopt_model(out,handle,status)
  end subroutine
  subroutine create_grid_model(depths,materials,interpolation,out,status,geometry,name)
    real(c_double), intent(in) :: depths(:)
    type(material), intent(in) :: materials(:)
    character(*), intent(in) :: interpolation
    type(model), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(model_geometry), optional, intent(in) :: geometry
    character(*), optional, intent(in) :: name
    character(:), allocatable :: actual_name
    type(c_geometry) :: geo
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    if(size(depths)/=size(materials)) then
      call invalid(status,'depths and materials must have equal lengths')
      return
    end if
    call pack_geometry(geometry,geo,status)
    if(.not.status%ok()) return
    actual_name=''
    if(present(name)) actual_name=name
    code=c_grid(size(depths,kind=c_size_t),depths,materials,trim(interpolation)//c_null_char, &
        geo,actual_name//c_null_char,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_model(out,handle,status)
  end subroutine

  subroutine compute_dispersion(input_model,period,out,status,wave,mode,outputs,options,context)
    type(model), intent(in) :: input_model
    real(c_double), intent(in) :: period
    type(mode_solution), intent(inout) :: out
    type(api_status), intent(out) :: status
    character(*), optional, intent(in) :: wave,outputs
    integer, optional, intent(in) :: mode
    type(computation_options), optional, intent(in) :: options
    type(dispersion_context), optional, intent(inout) :: context
    character(:), allocatable :: actual_wave,actual_outputs
    integer(c_int) :: actual_mode,code
    type(c_options) :: opts
    type(c_tracking_options) :: tracking
    type(c_ptr) :: handle,context_handle
    character(kind=c_char) :: msg(message_capacity)
    actual_wave='Rayleigh'
    actual_outputs='phase_velocity'
    actual_mode=0
    if(present(wave)) actual_wave=trim(wave)
    if(present(outputs)) actual_outputs=trim(outputs)
    if(present(mode)) actual_mode=mode
    opts=pack_options(options)
    tracking=pack_tracking_options(options,status)
    if(.not.status%ok()) return
    context_handle=c_null_ptr
    if(present(context)) then
      if(.not.context%valid()) then
        call invalid(status,'Empty dispersion context')
        return
      end if
      context_handle=context%handle
    end if
    code=c_dispersion_ex(input_model%handle,period,actual_wave//c_null_char,actual_mode, &
        actual_outputs//c_null_char,opts,tracking,context_handle,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_solution(out,handle,status)
  end subroutine

  subroutine compute_dispersion_curve(input_model,periods,out,status,wave,mode,outputs,options,context)
    type(model), intent(in) :: input_model
    real(c_double), intent(in) :: periods(:)
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    character(*), optional, intent(in) :: wave,outputs
    integer, optional, intent(in) :: mode
    type(computation_options), optional, intent(in) :: options
    type(dispersion_context), optional, intent(inout) :: context
    character(:), allocatable :: actual_wave,actual_outputs
    integer(c_int) :: actual_mode,code
    type(c_options) :: opts
    type(c_tracking_options) :: tracking
    type(c_ptr) :: handle,context_handle
    character(kind=c_char) :: msg(message_capacity)
    actual_wave='Rayleigh'
    actual_outputs='phase_velocity'
    actual_mode=0
    if(present(wave)) actual_wave=trim(wave)
    if(present(outputs)) actual_outputs=trim(outputs)
    if(present(mode)) actual_mode=mode
    opts=pack_options(options)
    tracking=pack_tracking_options(options,status)
    if(.not.status%ok()) return
    context_handle=c_null_ptr
    if(present(context)) then
      if(.not.context%valid()) then
        call invalid(status,'Empty dispersion context')
        return
      end if
      context_handle=context%handle
    end if
    code=c_curve_ex(input_model%handle,size(periods,kind=c_size_t),periods,actual_wave//c_null_char,actual_mode, &
        actual_outputs//c_null_char,opts,tracking,context_handle,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  subroutine model_info(this,out,status)
    class(model), intent(in) :: this
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_model_info(this%handle,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  ! Refinement levels are 0-based in every language; -1 selects the last saved level.
  subroutine compute_mode_count(solution,count,status,level)
    type(mode_solution), intent(in) :: solution
    integer, intent(out) :: count
    type(api_status), intent(out) :: status
    integer, intent(in), optional :: level
    integer(c_int) :: code,selected,value
    character(kind=c_char) :: msg(message_capacity)
    selected=-1_c_int
    if(present(level)) selected=int(level,c_int)
    code=c_mode_count(solution%handle,selected,value,msg,message_capacity)
    count=int(value)
    call finish_status(status,code,msg)
  end subroutine

  subroutine solution_info(this,out,status)
    class(mode_solution), intent(in) :: this
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_solution_info(this%handle,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  subroutine solution_model(this,out,status)
    class(mode_solution), intent(in) :: this
    type(model), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_solution_model(this%handle,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_model(out,handle,status)
  end subroutine

  subroutine solution_phase_velocity(this,value,status)
    class(mode_solution), intent(in) :: this
    real(c_double), intent(out) :: value
    type(api_status), intent(out) :: status
    type(result_data) :: info
    value=ieee_value(0.0_c_double,ieee_quiet_nan)
    call this%info(info,status)
    if(status%ok()) call info%scalar('phase_velocity',value,status)
  end subroutine

  subroutine solution_group_velocity(this,value,status)
    class(mode_solution), intent(in) :: this
    real(c_double), intent(out) :: value
    type(api_status), intent(out) :: status
    type(result_data) :: info
    value=ieee_value(0.0_c_double,ieee_quiet_nan)
    call this%info(info,status)
    if(status%ok()) call info%scalar('group_velocity',value,status)
  end subroutine

  subroutine solution_hv_ratio(this,value,status)
    class(mode_solution), intent(in) :: this
    real(c_double), intent(out) :: value
    type(api_status), intent(out) :: status
    type(result_data) :: info
    value=ieee_value(0.0_c_double,ieee_quiet_nan)
    call this%info(info,status)
    if(status%ok()) call info%scalar('hv_ratio',value,status)
  end subroutine

  subroutine compute_kernels(solution,basis,out,status,options,outputs)
    type(mode_solution), intent(in) :: solution
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(computation_options), optional, intent(in) :: options
    type(c_options) :: opts
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code

    character(*), intent(in) :: basis

    character(*), optional, intent(in) :: outputs
    character(:), allocatable :: actual_outputs

    actual_outputs='phase_velocity'
    if(present(outputs)) actual_outputs=trim(outputs)

    opts=pack_options(options)

    code=c_kernels(solution%handle,trim(basis)//c_null_char,actual_outputs//c_null_char, &
        opts,handle,msg,message_capacity)

    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  subroutine compute_hessian(solution,basis,out,status,options)
    type(mode_solution), intent(in) :: solution
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(computation_options), optional, intent(in) :: options
    type(c_options) :: opts
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code

    character(*), intent(in) :: basis

    opts=pack_options(options)

    code=c_hessian(solution%handle,trim(basis)//c_null_char, &
        opts,handle,msg,message_capacity)

    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  subroutine compute_interface_hessian_row(solution,interface_index,basis,out,status,options)
    type(mode_solution), intent(in) :: solution
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(computation_options), optional, intent(in) :: options
    type(c_options) :: opts
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code

    character(*), intent(in) :: basis

    integer, intent(in) :: interface_index

    if(interface_index<1) then
      call invalid(status,'Interface indices start at one in Fortran')
      return
    end if

    opts=pack_options(options)

    code=c_interface_hessian_row(solution%handle,int(interface_index-1,c_int),trim(basis)//c_null_char, &
        opts,handle,msg,message_capacity)

    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  subroutine compute_interface_kernels(solution,out,status,options)
    type(mode_solution), intent(in) :: solution
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(computation_options), optional, intent(in) :: options
    type(c_options) :: opts
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code

    opts=pack_options(options)

    code=c_interface_kernels(solution%handle, &
        opts,handle,msg,message_capacity)

    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  subroutine compute_energy_integrals(solution,out,status,options)
    type(mode_solution), intent(in) :: solution
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(computation_options), optional, intent(in) :: options
    type(c_options) :: opts
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code

    opts=pack_options(options)

    code=c_energy_integrals(solution%handle, &
        opts,handle,msg,message_capacity)

    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  subroutine compute_frequency_curvature(solution,out,status,options)
    type(mode_solution), intent(in) :: solution
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(computation_options), optional, intent(in) :: options
    type(c_options) :: opts
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code

    opts=pack_options(options)

    code=c_frequency_curvature(solution%handle, &
        opts,handle,msg,message_capacity)

    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine

  subroutine compute_hessian_reduced(solution,projection,out,status,options)
    type(mode_solution), intent(in) :: solution
    real(c_double), intent(in) :: projection(:,:)
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(computation_options), optional, intent(in) :: options
    type(c_options) :: opts
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    opts=pack_options(options)
    code=c_reduced(solution%handle,projection,size(projection,1,kind=c_size_t), &
        size(projection,2,kind=c_size_t),opts,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine
  subroutine compute_eigenfunctions(solution,out,status,num_sub_layers)
    type(mode_solution), intent(in) :: solution
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    integer, optional, intent(in) :: num_sub_layers
    integer(c_int) :: n,code
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    n=10
    if(present(num_sub_layers)) n=num_sub_layers
    code=c_eigenfunctions(solution%handle,n,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine
  subroutine compute_gravity_correction(solution,out,status,gravity,region,num_sub_layers,domega_relative)
    type(mode_solution), intent(in) :: solution
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    real(c_double), optional, intent(in) :: gravity,domega_relative
    character(*), optional, intent(in) :: region
    integer, optional, intent(in) :: num_sub_layers
    real(c_double) :: g,domega
    integer(c_int) :: n,code
    character(:), allocatable :: actual_region
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    g=9.80665e-3_c_double
    domega=5e-3_c_double
    n=40
    actual_region='solid'
    if(present(gravity)) g=gravity
    if(present(domega_relative)) domega=domega_relative
    if(present(num_sub_layers)) n=num_sub_layers
    if(present(region)) actual_region=trim(region)
    code=c_gravity(solution%handle,g,actual_region//c_null_char,n,domega,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine
  subroutine earth_flattening_transformation(input_model,out,status)
    type(model), intent(in) :: input_model
    type(model), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_earth_flattening_transformation(input_model%handle,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_model(out,handle,status)
  end subroutine
  subroutine apply_mineos_physical_dispersion(input_model,qkappa,qmu,reference_period,period,out,status)
    type(model), intent(in) :: input_model
    real(c_double), intent(in) :: qkappa(:),qmu(:),reference_period,period
    type(model), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    if(size(qkappa)/=size(qmu)) then
      call invalid(status,'qkappa and qmu must have equal lengths')
      return
    end if
    code=c_q(input_model%handle,size(qkappa,kind=c_size_t),qkappa,qmu,reference_period,period, &
        handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_model(out,handle,status)
  end subroutine
  subroutine get_capabilities(input_model,out,status,wave)
    type(model), intent(in) :: input_model
    type(result_data), intent(inout) :: out
    type(api_status), intent(out) :: status
    character(*), optional, intent(in) :: wave
    character(:), allocatable :: actual_wave
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    actual_wave='Rayleigh'
    if(present(wave)) actual_wave=trim(wave)
    code=c_capabilities(input_model%handle,actual_wave//c_null_char,handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_result(out,handle,status)
  end subroutine
  subroutine curve_solution(this,index,out,status)
    class(result_data), intent(in) :: this
    integer, intent(in) :: index
    type(mode_solution), intent(inout) :: out
    type(api_status), intent(out) :: status
    type(c_ptr) :: handle
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    if(index<1) then
      call invalid(status,'Curve indices start at one in Fortran')
      return
    end if
    code=c_curve_solution(this%handle,int(index-1,c_size_t),handle,msg,message_capacity)
    call finish_status(status,code,msg)
    call adopt_solution(out,handle,status)
  end subroutine
  subroutine result_numeric(this,field,values,status)
    class(result_data), intent(in) :: this
    character(*), intent(in) :: field
    real(c_double), allocatable, target, intent(out) :: values(:,:)
    type(api_status), intent(out) :: status
    integer(c_size_t) :: rows,columns,capacity
    character(kind=c_char) :: msg(message_capacity)
    integer(c_int) :: code
    code=c_numeric(this%handle,trim(field)//c_null_char,c_null_ptr,0_c_size_t,rows,columns,msg,message_capacity)
    call finish_status(status,code,msg)
    if(.not.status%ok()) return
    allocate(values(rows,columns))
    if(rows*columns==0) return
    capacity=rows*columns
    code=c_numeric(this%handle,trim(field)//c_null_char,c_loc(values),capacity,rows,columns,msg,message_capacity)
    call finish_status(status,code,msg)
    if(status%ok().and.trim(field)=='parameter.index') values=values+1
  end subroutine
  subroutine result_vector(this,field,values,status)
    class(result_data), intent(in) :: this
    character(*), intent(in) :: field
    real(c_double), allocatable, intent(out) :: values(:)
    type(api_status), intent(out) :: status
    real(c_double), allocatable :: matrix(:,:)
    call this%numeric(field,matrix,status)
    if(.not.status%ok()) return
    if(size(matrix,2)/=1) then
      call invalid(status,'Requested numeric field is not a vector')
      return
    end if
    values=matrix(:,1)
  end subroutine
  subroutine result_scalar(this,field,value,status)
    class(result_data), intent(in) :: this
    character(*), intent(in) :: field
    real(c_double), intent(out) :: value
    type(api_status), intent(out) :: status
    real(c_double), allocatable :: matrix(:,:)
    value=ieee_value(0.0_c_double,ieee_quiet_nan)
    call this%numeric(field,matrix,status)
    if(.not.status%ok()) return
    if(size(matrix)/=1) then
      call invalid(status,'Requested numeric field is not a scalar')
      return
    end if
    value=matrix(1,1)
  end subroutine
  subroutine result_text(this,field,value,status)
    class(result_data), intent(in) :: this
    character(*), intent(in) :: field
    character(:), allocatable, intent(out) :: value
    type(api_status), intent(out) :: status
    character(kind=c_char), allocatable, target :: buffer(:)
    character(kind=c_char) :: msg(message_capacity)
    integer(c_size_t) :: required,capacity
    integer(c_int) :: code
    integer :: i
    code=c_text(this%handle,trim(field)//c_null_char,c_null_ptr,0_c_size_t,required,msg,message_capacity)
    call finish_status(status,code,msg)
    if(.not.status%ok()) return
    capacity=required
    allocate(buffer(capacity))
    code=c_text(this%handle,trim(field)//c_null_char,c_loc(buffer),capacity,required,msg,message_capacity)
    call finish_status(status,code,msg)
    if(.not.status%ok()) return
    allocate(character(required-1)::value)
    do i=1,int(required)-1
      value(i:i)=buffer(i)
    end do
  end subroutine
  subroutine result_field_names(this,numeric,names,status)
    class(result_data), intent(in) :: this
    logical, intent(in) :: numeric
    character(:), allocatable, intent(out) :: names(:)
    type(api_status), intent(out) :: status
    character(kind=c_char), allocatable, target :: buffer(:)
    character(kind=c_char) :: msg(message_capacity)
    integer(c_size_t) :: count,required,capacity,i,max_length
    integer(c_int) :: code,kind
    integer :: j
    kind=merge(1,0,numeric)
    code=c_field_count(this%handle,kind,count,msg,message_capacity)
    call finish_status(status,code,msg)
    if(.not.status%ok()) return
    max_length=0
    do i=0,count-1
      code=c_field_name(this%handle,kind,i,c_null_ptr,0_c_size_t,required,msg,message_capacity)
      call finish_status(status,code,msg)
      if(.not.status%ok()) return
      max_length=max(max_length,required-1)
    end do
    allocate(character(max_length)::names(count))
    capacity=max_length+1
    allocate(buffer(capacity))
    names=''
    do i=0,count-1
      code=c_field_name(this%handle,kind,i,c_loc(buffer),capacity,required,msg,message_capacity)
      call finish_status(status,code,msg)
      if(.not.status%ok()) return
      do j=1,int(required)-1
        names(i+1)(j:j)=buffer(j)
      end do
    end do
  end subroutine
  subroutine result_parameter(this,index,entry,status)
    class(result_data), intent(in) :: this
    integer, intent(in) :: index
    type(parameter_axis_entry), intent(out) :: entry
    type(api_status), intent(out) :: status
    real(c_double), allocatable :: values(:)
    character(32) :: index_text
    character(:), allocatable :: prefix
    if(index<1) then
      call invalid(status,'Parameter-axis indices start at one in Fortran')
      return
    end if
    call this%vector('parameter.index',values,status)
    if(.not.status%ok()) return
    if(index>size(values)) then
      call invalid(status,'Parameter-axis index out of range')
      return
    end if
    entry%index=int(values(index))
    call this%vector('parameter.active',values,status)
    if(.not.status%ok()) return
    entry%active=int(values(index))/=0
    write(index_text,'(I0)') index-1
    prefix='parameter.'//trim(index_text)//'.'
    call this%text(prefix//'name',entry%name,status)
    if(.not.status%ok()) return
    call this%text(prefix//'entity',entry%entity,status)
    if(.not.status%ok()) return
    call this%text(prefix//'unit',entry%unit,status)
  end subroutine
end module disper26_api
