################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../blas/daxpy.o \
../blas/ddot.o \
../blas/dnrm2.o \
../blas/dscal.o 

C_SRCS += \
../blas/daxpy.c \
../blas/ddot.c \
../blas/dnrm2.c \
../blas/dscal.c 

OBJS += \
./blas/daxpy.o \
./blas/ddot.o \
./blas/dnrm2.o \
./blas/dscal.o 

C_DEPS += \
./blas/daxpy.d \
./blas/ddot.d \
./blas/dnrm2.d \
./blas/dscal.d 


# Each subdirectory must supply rules for building sources it contributes
blas/%.o: ../blas/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


