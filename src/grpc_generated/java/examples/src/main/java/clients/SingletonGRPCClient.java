package clients;

import java.nio.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import inference.GRPCInferenceServiceGrpc;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GrpcService.*;
import io.grpc.*;

import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContext;
import javax.net.ssl.SSLException;

// 单例模式的 gRPC 客户端类
public class SingletonGRPCClient {
    private static SingletonGRPCClient instance;
    private final ManagedChannel channel;
    private final GRPCInferenceServiceBlockingStub grpc_stub;

    // 私有构造函数，确保只能通过 getInstance 方法创建实例
    private SingletonGRPCClient(String host, int port) throws SSLException {
        SslContext sslContext = GrpcSslContexts.forClient().build();

        this.channel = NettyChannelBuilder.forAddress(host, port)
                .sslContext(sslContext)
                .build();
        this.grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);
    }

    // 获取单例实例的方法
    public static synchronized SingletonGRPCClient getInstance(String host, int port) throws SSLException {
        if (instance == null) {
            instance = new SingletonGRPCClient(host, port);
        }
        return instance;
    }

    // 泛型推理方法，允许用户动态指定输入输出的名称、数据类型和 shape
    public <T> Object infer(String modelName, String modelVersion,
                            String inputName, String inputDatatype, List<Integer> inputShape, List<T> inputData,
                            String outputName, String outputDataType) {
        // Generate the request
        ModelInferRequest.Builder request = ModelInferRequest.newBuilder();
        request.setModelName(modelName);
        request.setModelVersion(modelVersion);

        // Input data
        InferTensorContents.Builder inputDataBuilder = InferTensorContents.newBuilder();

        // 根据输入数据类型添加到输入张量中
        if (inputData.get(0) instanceof Float) {
            List<Float> floatData = inputData.stream().map(item -> (Float) item).collect(Collectors.toList());
            inputDataBuilder.addAllFp32Contents(floatData);
        } else if (inputData.get(0) instanceof Integer) {
            List<Integer> intData = inputData.stream().map(item -> (Integer) item).collect(Collectors.toList());
            inputDataBuilder.addAllIntContents(intData);
        } else if (inputData.get(0) instanceof Double) {
            List<Double> doubleData = inputData.stream().map(item -> (Double) item).collect(Collectors.toList());
            inputDataBuilder.addAllFp64Contents(doubleData);
        }
        // 可根据需要添加更多数据类型的处理

        // Populate the inputs in inference request
        ModelInferRequest.InferInputTensor.Builder input = ModelInferRequest.InferInputTensor.newBuilder();
        input.setName(inputName);
        input.setDatatype(inputDatatype);
        inputShape.forEach(input::addShape);
        input.setContents(inputDataBuilder);

        request.addInputs(0, input);

        // Populate the outputs in the inference request
        ModelInferRequest.InferRequestedOutputTensor.Builder output = ModelInferRequest.InferRequestedOutputTensor.newBuilder();
        output.setName(outputName);

        request.addOutputs(0, output);

        ModelInferResponse response = null;
        try {
            response = grpc_stub.withDeadlineAfter(1, TimeUnit.SECONDS).modelInfer(request.build());
        } catch (StatusRuntimeException e) {
            System.out.println(e.getMessage());
            return null;
        }

        // 根据输出数据类型创建对应类型的数组
        ByteBuffer outputBuffer = response.getRawOutputContentsList().get(0).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        if (outputDataType.equals("FP32")) {
            FloatBuffer floatBuffer = outputBuffer.asFloatBuffer();
            float[] result = new float[floatBuffer.remaining()];
            floatBuffer.get(result);
            return result;
        } else if (outputDataType.equals("INT32")) {
            IntBuffer intBuffer = outputBuffer.asIntBuffer();
            int[] result = new int[intBuffer.remaining()];
            intBuffer.get(result);
            return result;
        } else if (outputDataType.equals("FP64")) {
            DoubleBuffer doubleBuffer = outputBuffer.asDoubleBuffer();
            double[] result = new double[doubleBuffer.remaining()];
            doubleBuffer.get(result);
            return result;
        }
        // 可根据需要添加更多数据类型的处理

        return null;
    }

    // 关闭 channel 的方法
    public void shutdown() {
        channel.shutdownNow();
    }

    public static void main(String[] args) throws SSLException {
//        String host = args.length > 0 ? args[0] : "10.132.121.223";
        String host = args.length > 0 ? args[0] : "vas-vedap-model-infer-service-stg.nioint.com";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 443;

        String model_name = "driver";
        String model_version = "";

        // 获取单例实例
        SingletonGRPCClient client = SingletonGRPCClient.getInstance(host, port);

        // check server is live
        try {
            ServerLiveRequest serverLiveRequest = ServerLiveRequest.getDefaultInstance();
            ServerLiveResponse r = client.grpc_stub.serverLive(serverLiveRequest);
            System.out.println(r);
        } catch (StatusRuntimeException e) {
            System.out.println(e.getMessage());
            return;
        }

        // 生成随机输入数据（以 Float 为例）
        Random random = new Random();
        List<Float> inputData = DoubleStream.generate(() -> random.nextFloat() * 10).limit(2368).boxed().map(Float::new).collect(Collectors.toList());
//        List<Double> inputData = DoubleStream.generate(() -> random.nextFloat() * 10).limit(2368).boxed().collect(Collectors.toList());
        List<Integer> inputShape = Arrays.asList(32, 74);

        // 调用推理方法
        Object result = client.infer(model_name, model_version,
                "INPUT", "FP32", inputShape, inputData, "OUTPUT", "FP32");

        if (result instanceof float[]) {
            float[] floatResult = (float[]) result;
            for (int i = 0; i < floatResult.length; i++) {
                System.out.println(
                        inputData.get(i) + " = " + floatResult[i]);
            }
        } else {
            double[] doubleResult = (double[]) result;
            for (int i = 0; i < doubleResult.length; i++) {
                System.out.println(
                        inputData.get(i) + " = " + doubleResult[i]);
            }
        }

        // 关闭 channel
        client.shutdown();
    }
}