program test_api_fortran
  use disper26_api
  use, intrinsic :: iso_c_binding, only: c_sizeof
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  type(material) :: layer,halfspace,nodes(2)
  type(model) :: input_model,copied_model,grid,corrected
  type(model_geometry) :: geometry
  type(computation_options) :: options
  type(mode_solution) :: solution,copied_solution,selected
  type(result_data) :: info,copied_info,curve,derivatives
  type(parameter_axis_entry) :: parameter
  type(api_status) :: status
  real(c_double) :: c,u,c_repeat
  real(c_double) :: thicknesses(2),depths(2,2),periods(2,3),projection(12,4)
  integer :: policy,mode_count
  real(c_double), allocatable :: matrix(:,:),values(:),reference_hessian(:,:)
  character(:), allocatable :: text,names(:)

  call create_isotropic_parameters(3.0_c_double,5.2_c_double,2.5_c_double,layer,status)
  call success()
  call create_isotropic_parameters(4.5_c_double,8.0_c_double,3.3_c_double,halfspace,status)
  call success()
  if(c_sizeof(layer)/=9*c_sizeof(0.0_c_double)) error stop 'material C ABI size wrong'
  block
    type(material) :: water
    type(model) :: ocean
    type(result_data) :: metadata
    real(c_double), allocatable :: fields(:,:)
    call create_fluid_parameters(1.5_c_double,1.03_c_double,water,status)
    call success()
    water%gravity=0.00980665_c_double
    water%gravity_internal_scale=0.5_c_double
    water%gravity_surface_scale=0.25_c_double
    call create_layer_cake_model([water],[3.0_c_double],halfspace,ocean,status)
    call success()
    call ocean%info(metadata,status)
    call success()
    call metadata%numeric('materials',fields,status)
    call success()
    if(any(shape(fields)/=[9,1])) error stop 'fluid material matrix shape wrong'
    if(any(fields(7:9,1)/=[water%gravity,water%gravity_internal_scale,water%gravity_surface_scale])) &
      error stop 'fluid gravity fields changed across C ABI'
  end block
  block
    type(model) :: modern,legacy
    type(mode_solution) :: modern_solution,legacy_solution
    call create_layer_cake_model([layer],[10.0_c_double],halfspace,modern,status)
    call success()
    call create_pancake_model([layer],[10.0_c_double],halfspace,legacy,status)
    call success()
    call modern%info(info,status)
    call success()
    call info%text('kind',text,status)
    call success()
    if(text/='layer_cake') error stop 'new model kind lost'
    call legacy%info(info,status)
    call success()
    call info%text('kind',text,status)
    call success()
    if(text/='pancake') error stop 'legacy model kind lost'
    call compute_dispersion(modern,10.0_c_double,modern_solution,status)
    call success()
    call compute_dispersion(legacy,10.0_c_double,legacy_solution,status)
    call success()
    call modern_solution%phase_velocity(c,status)
    call success()
    call legacy_solution%phase_velocity(c_repeat,status)
    call success()
    if(c/=c_repeat) error stop 'legacy numerical result changed'
    call create_model([0.0_c_double,10.0_c_double],'pancake',legacy,status, &
      vs=[3.0_c_double,4.5_c_double],vp=[5.2_c_double,8.0_c_double],rho=[2.5_c_double,3.3_c_double])
    call success()
    call legacy%info(info,status)
    call success()
    call info%text('kind',text,status)
    call success()
    if(text/='pancake') error stop 'legacy array model kind lost'
  end block
  thicknesses=[10.0_c_double,-99.0_c_double]
  call create_layer_cake_model([layer],thicknesses(1:2:2),halfspace,input_model,status,name='Fortran test')
  call success()
  block
    type(dispersion_context) :: workspace,copy,empty
    type(mode_solution) :: saved,reused,fresh
    type(result_data) :: reused_curve
    real(c_double) :: before,after,reference
    call create_dispersion_context(workspace,status)
    call success()
    if(.not.workspace%valid()) error stop 'context creation failed'
    copy=workspace
    copy=copy
    call compute_dispersion(input_model,20.0_c_double,saved,status,context=workspace)
    call success()
    call saved%phase_velocity(before,status)
    call success()
    call compute_dispersion(input_model,10.0_c_double,reused,status,context=copy)
    call success()
    call compute_dispersion(input_model,10.0_c_double,fresh,status)
    call success()
    call reused%phase_velocity(after,status)
    call success()
    call fresh%phase_velocity(reference,status)
    call success()
    if(after/=reference) error stop 'Fortran context differs'
    call compute_dispersion_curve(input_model,[10.0_c_double,20.0_c_double],reused_curve,status,context=copy)
    call success()
    call compute_dispersion(input_model,-1.0_c_double,reused,status,context=copy)
    if(status%ok()) error stop 'invalid period accepted'
    call compute_dispersion(input_model,20.0_c_double,reused,status,context=copy)
    call success()
    call compute_dispersion(input_model,20.0_c_double,reused,status,context=empty)
    if(status%ok()) error stop 'empty context accepted'
    call reused%phase_velocity(after,status)
    call success()
    if(after/=before) error stop 'failed call changed output'
    workspace=empty
    copy=empty
    call saved%phase_velocity(after,status)
    call success()
    if(after/=before) error stop 'saved result changed after context release'
  end block
  block
    type(computation_options) :: recovery
    type(mode_solution) :: initial
    type(result_data) :: tracked,diagnostics
    real(c_double) :: verified
    recovery%method='continuation_recover'
    call compute_dispersion_curve(input_model,[8.0_c_double,8.2_c_double],tracked,status,options=recovery)
    call success()
    call tracked%get_solution(2,initial,status)
    call success()
    call initial%info(diagnostics,status)
    call success()
    call diagnostics%scalar('diagnostics.mode_verified',verified,status)
    call success()
    if(verified/=1) error stop 'Fortran recovery initial root unverified'
  end block
  copied_model=input_model
  block
    type(computation_options) :: controls
    type(dispersion_context) :: workspace
    type(mode_solution) :: seed,initial,missing
    type(result_data) :: tracked,metadata,fresh
    real(c_double) :: reference,counted,value
    real(c_double), allocatable :: reused_values(:),fresh_values(:)
    character(:), allocatable :: reason
    call compute_dispersion(input_model,8.2_c_double,seed,status)
    call success()
    call seed%phase_velocity(reference,status)
    call success()
    controls%verify='endpoints'
    controls%c_range=[reference*0.999_c_double,reference*1.001_c_double]
    controls%c_guess=1.0_c_double
    call create_dispersion_context(workspace,status)
    call success()
    call compute_dispersion_curve(input_model,[8.0_c_double,8.2_c_double],tracked,status, &
        options=controls,context=workspace)
    call success()
    call compute_dispersion_curve(input_model,[8.0_c_double,8.2_c_double],fresh,status,options=controls)
    call success()
    call tracked%vector('phase_velocity',reused_values,status)
    call success()
    call fresh%vector('phase_velocity',fresh_values,status)
    call success()
    if(any(reused_values/=fresh_values)) error stop 'extended Fortran context differs'
    call tracked%get_solution(2,initial,status)
    call success()
    call initial%info(metadata,status)
    call success()
    call metadata%scalar('diagnostics.mode_count_evaluations',counted,status)
    call success()
    if(counted/=2) error stop 'Fortran initial range did not use direct certification'
    call metadata%text('options.start',reason,status)
    call success()
    if(reason/='long') error stop 'Fortran default direction differs'
    call compute_dispersion(input_model,8.0_c_double,missing,status,mode=100,options=controls)
    call success()
    call missing%phase_velocity(value,status)
    call success()
    if(.not.ieee_is_nan(value)) error stop 'Fortran absent mode did not return missing'
    call missing%info(metadata,status)
    call success()
    call metadata%text('diagnostics.missing_reason',reason,status)
    call success()
    if(reason/='mode_not_found') error stop 'Fortran missing reason lost'
    call compute_mode_count(missing,mode_count,status)
    if(status%code/=API_CALCULATION_FAILED) error stop 'Fortran missing mode count accepted'
    controls%strict=.true.
    call compute_dispersion(input_model,8.0_c_double,missing,status,mode=100,options=controls)
    if(status%code/=API_CALCULATION_FAILED) error stop 'Fortran strict missing mode accepted'
    controls%c_range=[reference]
    call compute_dispersion(input_model,8.0_c_double,missing,status,options=controls)
    if(status%code/=API_INVALID_ARGUMENT) error stop 'Fortran malformed range accepted'
  end block
  copied_model=copied_model
  call copied_model%info(info,status)
  call success()
  call info%text('name',text,status)
  call success()
  if(text/='Fortran test') error stop 'model metadata missing'
  call compute_dispersion(copied_model,20.0_c_double,solution,status, &
      outputs='phase_velocity,group_velocity,hv_ratio')
  call success()
  call compute_mode_count(solution,mode_count,status)
  call success()
  if(mode_count/=0) error stop 'Fortran explicit mode count differs'
  call compute_mode_count(solution,mode_count,status,level=99)
  if(status%ok().or.mode_count/=-1) error stop 'Fortran invalid saved level accepted'
  call solution%phase_velocity(c,status)
  call success()
  call solution%group_velocity(u,status)
  call success()
  if(.not.ieee_is_finite(c).or..not.ieee_is_finite(u)) error stop 'invalid dispersion value'
  block
    real(c_double) :: z(2,2),vs(2),vp(2),rho(2)
    type(model) :: array_model
    type(mode_solution) :: array_solution
    type(material) :: vti_nodes(2)
    type(model) :: reference_model
    integer :: j
    real(c_double) :: reference_c
    z(1,:)=[0.0_c_double,10.0_c_double]
    z(2,:)=-999
    vs=[3.0_c_double,4.5_c_double];vp=[5.2_c_double,8.0_c_double];rho=[2.5_c_double,3.3_c_double]
    z(2,:)=vs
    call create_model(z(1,:),'layer_cake',array_model,status,vs=z(2,:),vp=vp,rho=rho)
    call success()
    call compute_dispersion(array_model,20.0_c_double,array_solution,status)
    call success()
    call array_solution%phase_velocity(c_repeat,status)
    call success()
    if(c_repeat/=c) error stop 'array constructor differs from object constructor'
    do j=1,2
      call create_velocity_vti_parameters(rho(j),vp(j)*1.02_c_double,vp(j), &
          vs(j)*1.03_c_double,vs(j),0.9_c_double,vti_nodes(j),status)
      call success()
    end do
    call create_layer_cake_model(vti_nodes(:1),[10.0_c_double],vti_nodes(2),reference_model,status)
    call success()
    call compute_dispersion(reference_model,20.0_c_double,array_solution,status)
    call success()
    call array_solution%phase_velocity(reference_c,status)
    call success()
    call create_model(z(1,:),'layer_cake',array_model,status,rho=rho, &
        vph=vp*1.02_c_double,vpv=vp,vsh=vs*1.03_c_double,vsv=vs,eta=[0.9_c_double,0.9_c_double])
    call success()
    call compute_dispersion(array_model,20.0_c_double,array_solution,status)
    call success()
    call array_solution%phase_velocity(c_repeat,status)
    call success()
    if(c_repeat/=reference_c) error stop 'VTI array component order wrong'
    call create_model(z(1,:),'layer_cake',array_model,status,rho=rho, &
        A=vti_nodes%A,C=vti_nodes%C,F=vti_nodes%F,L=vti_nodes%L,N=vti_nodes%N)
    call success()
    call compute_dispersion(array_model,20.0_c_double,array_solution,status)
    call success()
    call array_solution%phase_velocity(c_repeat,status)
    call success()
    if(c_repeat/=reference_c) error stop 'stiffness array component order wrong'
    call create_model(z(1,:),'grid',array_model,status,vs=vs,vp=vp,rho=rho,interpolation='velocity')
    call success()
    call create_model(z(1,:),'grid',array_model,status,vs=vs,vp=vp,rho=rho)
    if(status%ok()) error stop 'missing GRID interpolation accepted'
    call create_model(z(1,:),'layer_cake',array_model,status,vs=vs,rho=rho)
    if(status%ok()) error stop 'incomplete material arrays accepted'
    call create_model(z(1,:),'layer_cake',array_model,status,vs=vs,vp=vp,rho=rho,A=vp)
    if(status%ok()) error stop 'mixed material arrays accepted'
    call create_model([0.0_c_double,0.0_c_double],'layer_cake',array_model,status,vs=vs,vp=vp,rho=rho)
    if(status%ok()) error stop 'duplicate Layer-cake depths accepted'
    if(.not.array_model%valid()) error stop 'failed construction destroyed existing model'
  end block
  copied_solution=solution
  copied_solution=copied_solution
  ! Finalization of local copies must preserve surviving shared handles.
  block
    type(mode_solution) :: temporary
    type(model) :: temporary_model
    temporary=solution
    call temporary%get_model(temporary_model,status)
    call success()
  end block
  call copied_solution%phase_velocity(c_repeat,status)
  call success()
  if(c_repeat/=c) error stop 'solution ownership changed value'
  call solution%info(info,status)
  call success()
  copied_info=info
  copied_info=copied_info
  call copied_info%scalar('mode',c_repeat,status)
  call success()
  if(c_repeat/=0) error stop 'mode should remain zero based'
  call copied_info%field_names(.true.,names,status)
  call success()
  if(.not.any(names=='phase_velocity')) error stop 'field enumeration missing'

  periods(1,:)=[20.0_c_double,10.0_c_double,20.0_c_double]
  periods(2,:)=-999
  call compute_dispersion_curve(input_model,periods(1,:),curve,status)
  call success()
  call curve%vector('periods',values,status)
  call success()
  if(any(values/=periods(1,:))) error stop 'curve order or duplicate period lost'
  call curve%get_solution(3,selected,status)
  call success()
  call selected%phase_velocity(c_repeat,status)
  call success()
  ! Continuation may use a different bracket; allow four default root tolerances.
  if(abs(c_repeat-c)>4e-12_c_double*max(1.0_c_double,abs(c))) error stop 'curve solution mismatch'
  call selected%group_velocity(u,status)
  if(status%ok().or..not.ieee_is_nan(u)) error stop 'unrequested observable must be absent'

  call compute_kernels(solution,'stiffness_iso',derivatives,status)
  call success()
  call derivatives%parameter(1,parameter,status)
  call success()
  if(parameter%name/='Lambda') error stop 'incorrect isotropic stiffness axis'
  call compute_kernels(solution,'isotropic',derivatives,status)
  call success()
  call derivatives%vector('gradient.phase_velocity',values,status)
  call success()
  if(size(values)==0.or.any(.not.ieee_is_finite(values))) error stop 'invalid kernel'
  call derivatives%parameter(1,parameter,status)
  call success()
  if(parameter%index/=1) error stop 'parameter metadata must be one based'
  if(.not.parameter%active) error stop 'solid parameter incorrectly fixed'
  call derivatives%vector('parameter.index',values,status)
  call success()
  if(any(values<1)) error stop 'numeric parameter metadata must be one based'
  call compute_eigenfunctions(solution,info,status,num_sub_layers=4)
  call success()
  call info%vector('depth',values,status)
  call success()
  if(size(values)<2) error stop 'eigenfunction sampling missing'
  call info%field_names(.true.,names,status)
  call success()
  if(.not.any(index(names,'_real')>0)) error stop 'complex real fields missing'
  if(.not.any(index(names,'_imag')>0)) error stop 'complex imaginary fields missing'

  nodes=[layer,halfspace]
  depths(1,:)=[0.0_c_double,10.0_c_double]
  depths(2,:)=-999
  call create_grid_model(depths(1,:),nodes,'stiffness',grid,status)
  call success()
  call grid%info(info,status)
  call success()
  call info%numeric('materials',matrix,status)
  call success()
  if(any(shape(matrix)/=[9,2])) error stop 'nonsquare material matrix shape wrong'
  if(matrix(1,1)/=2.5_c_double.or.matrix(1,2)/=3.3_c_double) error stop 'column-major material layout wrong'
  if(matrix(2,2)/=halfspace%A) error stop 'material field order wrong'
  if(matrix(7,2)/=halfspace%gravity) error stop 'material gravity field order wrong'
  if(matrix(8,2)/=halfspace%gravity_internal_scale) error stop 'material internal scale field order wrong'
  if(matrix(9,2)/=halfspace%gravity_surface_scale) error stop 'material surface scale field order wrong'
  call info%vector('halfspace',values,status)
  call success()
  if(size(values)/=9) error stop 'halfspace material size wrong'
  if(values(9)/=halfspace%gravity_surface_scale) error stop 'halfspace surface scale field order wrong'
  options%max_steps=3
  options%max_levels=2
  call compute_dispersion(grid,20.0_c_double,selected,status,options=options)
  call success()
  projection=0
  projection(1,1)=1
  projection(6,1)=0.01_c_double
  projection(7,1)=0.2_c_double
  projection(4,3)=1
  projection(10,3)=0.4_c_double
  projection(:,2)=-999
  projection(:,4)=-999
  call compute_hessian_reduced(selected,projection(:,1:4:2),derivatives,status,options=options)
  call success()
  call derivatives%numeric('hessian',matrix,status)
  call success()
  if(any(shape(matrix)/=[2,2])) error stop 'reduced Hessian shape incorrect'
  call compute_hessian_reduced(selected,reshape([projection(:,1),projection(:,3)],[12,2]), &
      derivatives,status,options=options)
  call success()
  call derivatives%numeric('hessian',reference_hessian,status)
  call success()
  if(any(matrix/=reference_hessian)) error stop 'noncontiguous projection changed Hessian'
  call compute_frequency_curvature(solution,info,status)
  call success()
  call info%scalar('group_velocity_omega_derivative',u,status)
  call success()
  if(.not.ieee_is_finite(u)) error stop 'frequency curvature nonfinite'
  call compute_interface_kernels(solution,info,status)
  call success()
  call info%vector('gradient.phase_velocity',values,status)
  call success()
  if(size(values)/=1) error stop 'Layer-cake interface count incorrect'
  call apply_mineos_physical_dispersion(input_model,[1000.0_c_double,1000.0_c_double], &
      [600.0_c_double,600.0_c_double],1.0_c_double,20.0_c_double,corrected,status)
  call success()
  call corrected%info(info,status)
  call success()
  call info%scalar('q_period',u,status)
  call success()
  if(u/=20.0_c_double) error stop 'Q target period missing'
  do policy=1,2
    geometry%kind='spherical_ba97'
    if(policy==2) geometry%kind='spherical_experimental_density'
    geometry%flattening_exponent=-2.275_c_double
    call create_grid_model(depths(1,:),nodes,'stiffness',copied_model,status,geometry=geometry)
    call success()
    call earth_flattening_transformation(copied_model,corrected,status)
    call success()
    call corrected%info(info,status)
    call success()
    call info%text('flattening_policy',text,status)
    call success()
    if(text/=trim(geometry%kind)) error stop 'EFT policy metadata changed'
    call earth_flattening_transformation(corrected,copied_model,status)
    if(status%code/=API_INVALID_ARGUMENT) error stop 'Double EFT accepted'
  end do
  call create_grid_model([0.0_c_double,10.0_c_double,10.0_c_double,20.0_c_double], &
      [layer,layer,halfspace,halfspace],'stiffness',copied_model,status)
  call success()
  call compute_dispersion(copied_model,20.0_c_double,selected,status,options=options)
  call success()
  call compute_interface_hessian_row(selected,1,'stiffness',info,status,options=options)
  call success()
  call info%numeric('hessian',matrix,status)
  call success()
  if(any(shape(matrix)/=[1,25])) error stop 'interface Hessian row shape incorrect'
  call get_capabilities(grid,info,status)
  call success()
  call info%vector('supported',values,status)
  call success()
  if(size(values)==0) error stop 'missing capabilities'
  call create_grid_model(depths(1,:),nodes(:1),'stiffness',grid,status)
  if(status%code/=API_INVALID_ARGUMENT) error stop 'shape mismatch not rejected'
  if(.not.grid%valid()) error stop 'failed update must preserve existing model'
  call curve%get_solution(0,selected,status)
  if(status%code/=API_INVALID_ARGUMENT) error stop 'zero Fortran curve index accepted'
  call info%scalar('missing_field',u,status)
  if(status%ok().or..not.ieee_is_nan(u)) error stop 'unknown field appeared as zero'
  print *, 'Common Fortran API lifetime, noncontiguous arrays, metadata and errors passed'
contains
  subroutine success()
    if(.not.status%ok()) then
      print *, status%code,': ',status%message
      error stop 'unexpected API error'
    end if
  end subroutine
end program test_api_fortran
