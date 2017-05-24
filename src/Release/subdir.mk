################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Album.cpp \
../PersonClassifier.cpp \
../Photo.cpp \
../Solver.cpp \
../Utils.cpp \
../config.cpp \
../linear.cpp \
../main.cpp \
../train.cpp \
../tron.cpp 

OBJS += \
./Album.o \
./PersonClassifier.o \
./Photo.o \
./Solver.o \
./Utils.o \
./config.o \
./linear.o \
./main.o \
./train.o \
./tron.o 

CPP_DEPS += \
./Album.d \
./PersonClassifier.d \
./Photo.d \
./Solver.d \
./Utils.d \
./config.d \
./linear.d \
./main.d \
./train.d \
./tron.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -I/usr/local/include -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


