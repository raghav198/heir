// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s

#encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=15>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#rlwe_params = #lwe.rlwe_params<dimension=2, ring=#ring>
!plaintext_rlwe = !lwe.rlwe_plaintext<encoding = #encoding, ring = #ring, underlying_type=i3>
!ciphertext_rlwe = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #rlwe_params, underlying_type=i3>
!rlwe_key = !lwe.rlwe_secret_key<rlwe_params=#rlwe_params>

func.func @test_rlwe_sk_encrypt(%arg0: !plaintext_rlwe, %arg1: !rlwe_key) -> !ciphertext_rlwe {
  // CHECK-NOT: lwe.rlwe_encrypt

  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index

  // CHECK-DAG: %[[PRNG:.*]] = random.init_prng %[[SEED:.*]]

  // CHECK-DAG: %[[UNIFORM:.*]] = random.discrete_uniform_distribution %[[PRNG]]
  // CHECK-DAG: %[[SAMPLE_U:.*]] = random.sample %[[UNIFORM]]
  // CHECK-DAG: %[[U:.*]] = polynomial.from_tensor %[[SAMPLE_U]]

  // CHECK-DAG: %[[GAUSSIAN:.*]] = random.discrete_gaussian_distribution %[[PRNG]]
  // CHECK-DAG: %[[SAMPLE_E:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK-DAG: %[[E:.*]] = polynomial.from_tensor %[[SAMPLE_E]]

  // CHECK-DAG: %[[SK:.*]] = tensor.extract %arg1[%[[ZERO]]]

  // CHECK: %[[U_TIMES_SK:.*]] = polynomial.mul %[[U]], %[[SK]]
  // CHECK: %[[U_TIMES_SK_PLUS_M:.*]] = polynomial.add %[[U_TIMES_SK]], %arg0
  // CHECK: %[[C_1:.*]] = polynomial.add %[[U_TIMES_SK_PLUS_M]], %[[E]]

  // CHECK:     %[[C:.*]] = tensor.from_elements %[[U]], %[[C_1]]
  // CHECK:     return %[[C]]
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!plaintext_rlwe, !rlwe_key) -> !ciphertext_rlwe
  return %0 : !ciphertext_rlwe
}
