#pragma once

// An experimental enum which holds offsets for grid quantities
// In the final form of this approach, this file will also set nfields and NSCALARS, 
// so that adding a field only requires registering it here.


// enum notes:
// Must be "unscoped" to be implicitly treated as int: this means cannot use "enum class" or "enum struct"
// Wrapped in namespace to give it an effective scope to prevent collisions
// enum values (i.e. density) belong to their enclosing scope, which necessitates the namespace wrapping
// --otherwise "density" would be available in global scope
// ": int" forces underlying type to be int

namespace grid_enum {
enum : int {

  // Don't touch hydro quantities until all of hydro is made consistent with grid_enum (if ever)
  density,
  momentum_x,
  momentum_y,
  momentum_z,
  Energy,

  // Code assumes scalars are a contiguous block
  #ifdef SCALAR
  scalar,
  scalar_minus_1 = scalar - 1,// so that next enum item starts at same index as scalar

  // Add scalars here, wrapped appropriately with ifdefs:
  #ifdef BASIC_SCALAR
  basic_scalar,
  #endif

  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  HI_density,
  HII_density,
  HeI_density,
  HeII_density,
  HeIII_density,
  e_density,
  #ifdef GRACKLE_METALS
  metal_density,
  #endif
  #endif


  finalscalar_plus_1, // needed to calculate NSCALARS
  finalscalar = finalscalar_plus_1 - 1; // resets enum to finalscalar so fields afterwards are correct
  
  // so that anything after starts with scalar + NSCALARS
  #endif // SCALAR
  #ifdef MHD
  magnetic_x,
  magnetic_y,
  magnetic_z,
  #endif
  #ifdef DE
  GasEnergy,
  #endif
  num_fields,

//Aliases
  nscalars = finalscalar_plus_1 - scalar,

};
}
