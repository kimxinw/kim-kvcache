#include "heteropage_kv/reference/kv_reference.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace kimkvcache;

int failures = 0;

void expect(bool condition, std::string const& message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

void testHalfConversion()
{
    expect(floatToKvScalar(0.0F) == 0x0000U, "+0 encoding");
    expect(floatToKvScalar(-0.0F) == 0x8000U, "-0 encoding");
    expect(floatToKvScalar(1.0F) == 0x3C00U, "one encoding");
    expect(floatToKvScalar(-2.0F) == 0xC000U, "minus two encoding");
    expect(floatToKvScalar(65504.0F) == 0x7BFFU, "maximum half encoding");

    expect(kvScalarToFloat(0x3C00U) == 1.0F, "one decoding");
    expect(kvScalarToFloat(0xC000U) == -2.0F, "minus two decoding");
    expect(kvScalarToFloat(0x7BFFU) == 65504.0F, "maximum half decoding");
    expect(
        std::isinf(kvScalarToFloat(0x7C00U)),
        "infinity decoding"
    );
    expect(
        std::isnan(kvScalarToFloat(0x7E00U)),
        "NaN decoding"
    );
}

void testReferenceAttention()
{
    KvLayout const layout{1, 1, 2};
    std::vector<KvScalar> kv(8, floatToKvScalar(0.0F));

    kv[layout.offset(0, KvComponent::Key, 0, 0, 0, 2)] =
        floatToKvScalar(0.0F);
    kv[layout.offset(0, KvComponent::Key, 0, 0, 1, 2)] =
        floatToKvScalar(0.0F);
    kv[layout.offset(0, KvComponent::Key, 1, 0, 0, 2)] =
        floatToKvScalar(1.0F);
    kv[layout.offset(0, KvComponent::Key, 1, 0, 1, 2)] =
        floatToKvScalar(0.0F);

    kv[layout.offset(0, KvComponent::Value, 0, 0, 0, 2)] =
        floatToKvScalar(1.0F);
    kv[layout.offset(0, KvComponent::Value, 0, 0, 1, 2)] =
        floatToKvScalar(2.0F);
    kv[layout.offset(0, KvComponent::Value, 1, 0, 0, 2)] =
        floatToKvScalar(3.0F);
    kv[layout.offset(0, KvComponent::Value, 1, 0, 1, 2)] =
        floatToKvScalar(4.0F);

    std::vector<float> const query{1.0F, 0.0F};
    std::vector<float> output;
    expect(
        referenceAttention(layout, kv, 2, query, output),
        "valid Reference Attention"
    );

    float const second_weight = std::exp(1.0F / std::sqrt(2.0F));
    float const denominator = 1.0F + second_weight;
    float const expected0 = (1.0F + 3.0F * second_weight) / denominator;
    float const expected1 = (2.0F + 4.0F * second_weight) / denominator;
    expect(output.size() == 2, "Reference Attention output size");
    if (output.size() == 2) {
        expect(std::abs(output[0] - expected0) < 1.0e-6F, "output dim 0");
        expect(std::abs(output[1] - expected1) < 1.0e-6F, "output dim 1");
    }

    std::vector<float> invalid_output;
    expect(
        !referenceAttention(layout, kv, 0, query, invalid_output),
        "zero tokens rejected"
    );
    expect(
        !referenceAttention(layout, kv, 2, std::vector<float>{1.0F},
            invalid_output),
        "invalid query shape rejected"
    );
}

} // namespace

int main()
{
    testHalfConversion();
    testReferenceAttention();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All contiguous Reference/K4 tests passed\n";
    return 0;
}
